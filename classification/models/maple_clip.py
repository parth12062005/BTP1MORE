"""
MaPLe CLIP model for test-time adaptation.
Adapted from multimodal-prompt-learning to work with classification directory structure.
Uses layer-wise deep prompt insertion via VisionTransformer_MaPLe and MaPLe-specific text encoder.
"""

import sys
import os
import copy
import torch
import torch.nn as nn
import logging
from collections import OrderedDict
from typing import List

# Add multimodal-prompt-learning to path to import MaPLe components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../multimodal-prompt-learning'))
from clip.model import (
    ResidualAttentionBlock_MaPLe,
    VisionTransformer_MaPLe,
    Transformer,
    LayerNorm,
    build_model as build_maple_model,
)
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from open_clip import get_tokenizer
from datasets.cls_names import get_class_names

logger = logging.getLogger(__name__)
_tokenizer = _Tokenizer()


def _get_clones(module, N):
    """Clone a module N times."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MaPLeTextEncoder(nn.Module):
    """
    Text encoder for MaPLe that handles deep prompts layer-wise.
    Similar to multimodal-prompt-learning/trainers/maple.py TextEncoder.
    """
    def __init__(self, clip_model, maple_transformer=None):
        super().__init__()
        # Use provided MaPLe transformer or create one
        if maple_transformer is not None:
            self.transformer = maple_transformer
        else:
            # Build MaPLe-compatible transformer
            transformer_width = clip_model.ln_final.weight.shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(set(k.split(".")[2] for k in clip_model.state_dict().keys() 
                                        if k.startswith("transformer.resblocks")))
            
            design_details = {
                "trainer": "MaPLe",
                "vision_depth": 0,
                "language_depth": 0,
                "vision_ctx": 0,
                "language_ctx": 0,
                "maple_length": 4,  # Will be set properly in forward
            }
            
            # Create MaPLe transformer
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=clip_model.attn_mask,
                prompts_needed=0,
                text_layer=True,
                design_details=design_details,
            )
            
            # Copy weights from original transformer to MaPLe transformer
            self._copy_transformer_weights(clip_model.transformer)
        
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.visual.conv1.weight.dtype
    
    def _copy_transformer_weights(self, original_transformer):
        """Copy weights from original transformer to MaPLe transformer."""
        for orig_block, maple_block in zip(original_transformer.resblocks, self.transformer.resblocks):
            maple_block.attn.load_state_dict(orig_block.attn.state_dict())
            maple_block.ln_1.load_state_dict(orig_block.ln_1.state_dict())
            maple_block.ln_2.load_state_dict(orig_block.ln_2.state_dict())
            maple_block.mlp.load_state_dict(orig_block.mlp.state_dict())
    
    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        """
        Forward pass with deep prompts inserted layer-wise.
        
        Args:
            prompts: (n_cls, seq_len, dim) text prompts with shallow context
            tokenized_prompts: (n_cls, max_seq_len) tokenized prompts
            compound_prompts_deeper_text: List of (n_ctx, dim) deep prompt tensors
        """
        x = prompts + self.positional_embedding[:prompts.shape[1], :].type(self.dtype).unsqueeze(0)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Pass as list for nn.Sequential compatibility
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is counter
        outputs = self.transformer(combined)
        x = outputs[0]  # extract x back
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # Take features from EOT token
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class MaPLePromptLearner(nn.Module):
    """
    MaPLe prompt learner with shallow + deep compound prompts.
    Adapted from multimodal-prompt-learning/trainers/maple.py MultiModalPromptLearner.
    """
    def __init__(
        self,
        clip_model,
        arch_name,
        class_names,
        n_ctx: int = 4,
        ctx_init: str = None,
        prompt_depth: int = 2,
        class_token_pos: str = "end",
    ):
        super().__init__()
        self.n_cls = len(class_names)
        self.n_ctx = n_ctx
        self.prompt_depth = prompt_depth
        self.class_token_position = class_token_pos
        
        assert prompt_depth >= 1, "MaPLe prompt_depth must be >= 1"
        
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.device = clip_model.visual.conv1.weight.device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.token_embedding = clip_model.token_embedding
        
        # Convert arch_name format (ViT-B/16 -> ViT-B-16) for get_tokenizer
        tokenizer_arch = arch_name.replace('/', '-') if '/' in arch_name else arch_name
        self.tokenize = get_tokenizer(tokenizer_arch)
        
        # Initialize shallow prompts
        if ctx_init and n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        logger.info('MaPLe design: Multi-modal Prompt Learning')
        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of MaPLe context words (tokens): {n_ctx}")
        
        # Shallow prompts
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Projection from text ctx_dim -> visual embedding dim
        vis_dim = clip_model.visual.conv1.weight.shape[0]
        self.proj = nn.Linear(self.ctx_dim, vis_dim)
        self.proj.half()
        
        # Deep compound prompts for text (prompt_depth - 1 layers)
        self.compound_prompts_text = nn.ParameterList([
            nn.Parameter(torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype))
            for _ in range(self.prompt_depth - 1)
        ])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        
        # Projection layers for each deep prompt (text -> visual)
        single_layer = nn.Linear(self.ctx_dim, vis_dim)
        self.compound_prompt_projections = _get_clones(single_layer, self.prompt_depth - 1)
        for proj in self.compound_prompt_projections:
            proj.half()
        
        # Setup tokenized prompts
        class_names = [name.replace("_", " ") for name in class_names]
        name_lens = [len(self.tokenize(name)) for name in class_names]
        prompts = [prompt_prefix + " " + name + "." for name in class_names]
        
        tokenized_prompts = torch.cat([self.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
    
    def construct_prompts(self, ctx, prefix, suffix):
        """Construct prompts from context, prefix, and suffix."""
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts
    
    def forward(self):
        """
        Forward pass returning prompts and deep prompts.
        Returns:
            prompts: (n_cls, seq_len, ctx_dim) text prompts with shallow context
            shared_ctx: (n_ctx, vis_dim) shallow visual prompts
            deep_compound_prompts_text: List of (n_ctx, ctx_dim) deep text prompts
            deep_compound_prompts_vision: List of (n_ctx, vis_dim) deep visual prompts
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        
        # Shallow visual prompts
        shared_ctx = self.proj(self.ctx)  # (n_ctx, vis_dim)
        
        # Deep compound prompts
        deep_compound_prompts_text = list(self.compound_prompts_text)
        
        # Project deep text prompts to visual space
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        
        return prompts, shared_ctx, deep_compound_prompts_text, visual_deep_prompts


class ClipMaPLe(nn.Module):
    """
    MaPLe-style CLIP wrapper for test-time adaptation.
    
    Uses VisionTransformer_MaPLe for image encoder and MaPLeTextEncoder for text encoder.
    Deep prompts are inserted layer-wise in both encoders.
    """
    def __init__(
        self,
        clip_model,
        normalization,
        arch_name,
        dataset_name,
        n_ctx: int = 4,
        prompt_depth: int = 2,
        ctx_init: str = None,
        class_token_pos: str = "end",
    ):
        super().__init__()
        self.clip_model = clip_model
        self.normalize = normalization
        
        class_names = get_class_names(dataset_name)
        
        # Build MaPLe CLIP model using build_model from multimodal-prompt-learning
        # This ensures proper MaPLe architecture with layer-wise prompt insertion
        design_details = {
            "trainer": "MaPLe",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
            "maple_length": n_ctx,
        }
        
        # Get state dict from original model
        state_dict = clip_model.state_dict()
        
        # Build MaPLe CLIP model
        maple_clip_model = build_maple_model(state_dict, design_details)
        
        # Extract MaPLe visual encoder (VisionTransformer_MaPLe)
        self.image_encoder = maple_clip_model.visual
        
        # Prompt learner
        self.prompt_learner = MaPLePromptLearner(
            clip_model,
            arch_name,
            class_names,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            prompt_depth=prompt_depth,
            class_token_pos=class_token_pos,
        )
        
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        # Text encoder - use MaPLe transformer from maple_clip_model
        self.text_encoder = MaPLeTextEncoder(clip_model, maple_transformer=maple_clip_model.transformer)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.visual.conv1.weight.dtype
    
    def forward(self, image, return_features: bool = False):
        """
        Forward pass with MaPLe architecture.
        
        Args:
            image: (B, 3, H, W) input images
            return_features: If True, return features for BATCLIP losses
        
        Returns:
            If return_features=False: logits (B, n_cls)
            If return_features=True: (logits, image_features, text_features_flat, img_pre_features, text_pre_features)
        """
        # Normalize input
        image = self.normalize(image.type(self.dtype))
        
        # Get prompts from prompt learner
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        
        # Encode text with deep prompts
        text_pre_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text)
        text_features = text_pre_features / (text_pre_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Encode image with deep prompts
        img_pre_features = self.image_encoder(image, shared_ctx, deep_compound_prompts_vision)
        image_features = img_pre_features / (img_pre_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute logits
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (image_features @ text_features.t())
        
        if not return_features:
            return logits
        
        # Return BATCLIP-compatible outputs
        text_features_flat = text_features  # (n_cls, dim), normalized
        return logits, image_features, text_features_flat, img_pre_features, text_pre_features
