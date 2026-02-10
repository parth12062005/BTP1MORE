"""
Adapted from: https://github.com/azshue/TPT/blob/main/clip/custom_clip.py
Paper: https://arxiv.org/pdf/2209.07511.pdf
"""

import torch
import torch.nn as nn
import logging
from collections import OrderedDict

from open_clip import create_model_and_transforms, get_tokenizer
from datasets.cls_names import get_class_names

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.attn_mask = clip_model.attn_mask

    def forward(self, prompts, tokenized_prompts):
        # prompts: (N, L, dim) where L is the sequence length (can vary)
        # positional_embedding: (max_seq_len, dim), need to slice to match L
        seq_len = prompts.shape[1]
        pos_emb = self.positional_embedding[:seq_len, :].type(self.dtype)
        x = prompts + pos_emb.unsqueeze(0)  # (N, L, dim)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Handle attention mask for variable sequence lengths
        # If attn_mask exists and is larger than seq_len, slice it
        if self.attn_mask is not None:
            if self.attn_mask.shape[0] > seq_len:
                attn_mask = self.attn_mask[:seq_len, :seq_len]
            else:
                attn_mask = self.attn_mask
        else:
            attn_mask = None
        
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, arch_name, class_names, n_ctx=16, ctx_init=None, class_token_pos='end', learned_cls=False):
        super().__init__()
        self.n_cls = len(class_names)
        self.learned_cls = learned_cls
        self.class_names = class_names

        self.dtype = clip_model.visual.conv1.weight.dtype
        self.device = clip_model.visual.conv1.weight.device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.token_embedding = clip_model.token_embedding
        # Convert arch_name format (ViT-B/16 -> ViT-B-16) for get_tokenizer
        tokenizer_arch = arch_name.replace('/', '-') if '/' in arch_name else arch_name
        self.tokenize = get_tokenizer(tokenizer_arch)

        if ctx_init:
            # use given words to initialize context vectors
            logger.info("Initializing the context with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                class_token_pos = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = self.tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            logger.info("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.n_ctx = n_ctx
        self.prompt_prefix = prompt_prefix
        self.class_token_position = class_token_pos

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {self.n_ctx}")

        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # setup the rest using the specified class names
        self.reset_class_names(class_names)

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)  # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_class_names(self, class_names):
        self.n_cls = len(class_names)
        if not self.learned_cls:
            class_names = [name.replace("_", " ") for name in class_names]
            name_lens = [len(self.tokenize(name)) for name in class_names]
            prompts = [self.prompt_prefix + " " + name + "." for name in class_names]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in class_names]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in class_names]

            # TODO: re-init the cls parameters
            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        with torch.no_grad():
            tokenized_prompts = torch.cat([self.tokenize(p) for p in prompts]).to(self.device)
            embedding = self.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.class_names = class_names

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        ctx = init if init is not None else self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.learned_cls:
            assert self.class_token_position == "end"

        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            if self.split_idx is not None:
                half_n_ctx = self.split_idx  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError(f"Class token position '{self.class_token_position}' is not supported."
                             f" Choose from: end, middle, front")

        return prompts


class CoCoOpPromptLearner(PromptLearner):
    """
    Image-conditioned prompt learner for CoCoOp.
    One shared bias π per image: v_m(x) = v_m + π.
    Meta-net: 3-layer MLP with LayerNorm: Linear -> LN -> ReLU -> Linear -> LN -> ReLU -> Linear.
    """
    def __init__(self, clip_model, arch_name, class_names, n_ctx=16, ctx_init=None, class_token_pos='end', learned_cls=False):
        super().__init__(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)
        
        ctx_dim = self.ctx_dim
        vis_dim = ctx_dim
        h1, h2 = vis_dim // 4, vis_dim // 16
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, h1)),
            ("norm1", nn.LayerNorm(h1)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(h1, h2)),
            ("norm2", nn.LayerNorm(h2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("linear3", nn.Linear(h2, ctx_dim)),
        ]))
    
    def forward(self, image_features=None, init=None):
        """
        Forward pass for CoCoOp prompt learner.
        
        Args:
            image_features: (B, dim) or (dim,) image features from encoder
            init: Optional initial ctx vectors (for reset)
        
        Returns:
            prompts: (B, n_cls, n_tok, dim) or (n_cls, n_tok, dim) prompt tensors
        """
        if image_features is None:
            # Fallback to base class behavior (static prompts)
            return super().forward(init=init)
        
        # Handle both batched and unbatched inputs
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B = image_features.shape[0]
        
        # Ensure same dtype as meta_net (image_features may be half when CLIP uses fp16)
        meta_dtype = next(self.meta_net.parameters()).dtype
        image_features = image_features.to(meta_dtype)
        
        # Same as external CoCoOp: one bias π per image, added to all context tokens (v_m(x) = v_m + π)
        bias = self.meta_net(image_features)  # (B, ctx_dim)
        bias = bias.unsqueeze(1)  # (B, 1, ctx_dim)
        base_ctx = init if init is not None else self.ctx  # (n_ctx, ctx_dim)
        ctx = base_ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx = ctx + bias  # (B, n_ctx, ctx_dim)
        # Cast to model dtype so concat with prefix/suffix (from token_embedding) matches
        ctx = ctx.to(self.dtype)
        
        # Expand ctx to (B, n_cls, n_ctx, ctx_dim)
        ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        
        # Get prefix and suffix
        prefix = self.token_prefix  # (n_cls, 1, dim)
        suffix = self.token_suffix  # (n_cls, *, dim)
        
        # Expand prefix and suffix to batch dimension
        prefix = prefix.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_cls, 1, dim)
        suffix = suffix.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_cls, *, dim)
        
        if self.learned_cls:
            assert self.class_token_position == "end"
            cls = self.cls.unsqueeze(0).expand(B, -1, -1, -1)  # (B, n_cls, 1, dim)
        
        # Construct prompts based on class token position
        if self.class_token_position == "end":
            if self.learned_cls:
                prompts = torch.cat([prefix, ctx, cls, suffix], dim=-2)  # (B, n_cls, n_tok, dim)
            else:
                prompts = torch.cat([prefix, ctx, suffix], dim=-2)  # (B, n_cls, n_tok, dim)
        elif self.class_token_position == "middle":
            half_n_ctx = self.split_idx if self.split_idx is not None else self.n_ctx // 2
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:, i:i+1, :, :]  # (B, 1, 1, dim)
                class_i = suffix[:, i:i+1, :name_len, :]  # (B, 1, name_len, dim)
                suffix_i = suffix[:, i:i+1, name_len:, :]  # (B, 1, *, dim)
                ctx_i_half1 = ctx[:, i:i+1, :half_n_ctx, :]  # (B, 1, n_ctx//2, dim)
                ctx_i_half2 = ctx[:, i:i+1, half_n_ctx:, :]  # (B, 1, n_ctx//2, dim)
                prompt_i = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=-2)
                prompts_list.append(prompt_i)
            prompts = torch.cat(prompts_list, dim=1)  # (B, n_cls, n_tok, dim)
        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[:, i:i+1, :, :]
                class_i = suffix[:, i:i+1, :name_len, :]
                suffix_i = suffix[:, i:i+1, name_len:, :]
                ctx_i = ctx[:, i:i+1, :, :]
                prompt_i = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=-2)
                prompts_list.append(prompt_i)
            prompts = torch.cat(prompts_list, dim=1)  # (B, n_cls, n_tok, dim)
        else:
            raise ValueError(f"Class token position '{self.class_token_position}' is not supported.")
        
        if squeeze_output:
            prompts = prompts.squeeze(0)  # (n_cls, n_tok, dim)
        
        return prompts


class FusionMLP(nn.Module):
    """
    Fusion of image and text encodings: (img_enc, text_enc) -> 1024 -> 512 -> 128.
    BN between every two layers; BNs are adapted during TTA.
    """
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, hidden_dim)),
            ("bn1", nn.BatchNorm1d(hidden_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(hidden_dim, output_dim)),
            ("bn2", nn.BatchNorm1d(output_dim)),
        ]))

    def forward(self, x):
        return self.net(x)


class BiasHead(nn.Module):
    """
    Head from fusion output 128 -> 64 -> bias (embed_dim). BN between layers; adapted at TTA.
    Used for text-bias (CoCoOp-type) and image-bias (reverse CoCoOp-type).
    """
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, hidden_dim)),
            ("bn1", nn.BatchNorm1d(hidden_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(hidden_dim, output_dim)),
        ]))

    def forward(self, x):
        return self.net(x)


class VisualContextPromptLearner(nn.Module):
    """
    CoCoOp-style visual context prompt learner for image side.
    Learns visual context tokens [C1] [C2] [C3] that are inserted before CLS in ViT.
    Meta-net adds conditional bias based on text features (text conditions image).
    """
    def __init__(self, clip_model, n_ctx_vis=4, ctx_init=None):
        super().__init__()
        self.n_ctx_vis = n_ctx_vis
        vis_dim = clip_model.visual.conv1.weight.shape[0]  # patch embedding dim (e.g., 512 for ViT-B)
        self.dtype = clip_model.visual.conv1.weight.dtype
        self.device = clip_model.visual.conv1.weight.device
        
        # Learnable visual context tokens (same dim as patches)
        if ctx_init:
            # Initialize from text (not typical, but allows initialization)
            logger.info("Visual context init not implemented yet; using random init")
        ctx_vectors = torch.empty(n_ctx_vis, vis_dim, dtype=self.dtype, device=self.device)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # (n_ctx_vis, vis_dim)
        
        # Meta-net: text features -> bias for visual context tokens
        # Input: text embedding dim (ctx_dim), output: vis_dim (for each ctx token)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        h1, h2 = ctx_dim // 4, ctx_dim // 16
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(ctx_dim, h1)),
            ("norm1", nn.LayerNorm(h1)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(h1, h2)),
            ("norm2", nn.LayerNorm(h2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("linear3", nn.Linear(h2, vis_dim)),
        ]))
    
    def reset(self):
        """Reset visual context tokens to initial state."""
        self.ctx.copy_(self.ctx_init_state)
    
    def forward(self, text_features=None):
        """
        Forward pass for visual context prompt learner.
        
        Args:
            text_features: (B, dim) or (dim,) text features (for conditional bias)
        
        Returns:
            visual_ctx: (B, n_ctx_vis, vis_dim) visual context tokens with conditional bias
        """
        if text_features is None:
            # Static visual context (no conditional bias)
            if self.ctx.dim() == 2:
                return self.ctx.unsqueeze(0)  # (1, n_ctx_vis, vis_dim)
            return self.ctx
        
        # Handle both batched and unbatched inputs
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B = text_features.shape[0]
        
        # Meta-net: text -> bias for visual context
        meta_dtype = next(self.meta_net.parameters()).dtype
        text_features = text_features.to(meta_dtype)
        bias = self.meta_net(text_features)  # (B, vis_dim)
        bias = bias.unsqueeze(1)  # (B, 1, vis_dim)
        
        # Add bias to base visual context tokens
        base_ctx = self.ctx  # (n_ctx_vis, vis_dim)
        ctx = base_ctx.unsqueeze(0)  # (1, n_ctx_vis, vis_dim)
        ctx = ctx + bias  # (B, n_ctx_vis, vis_dim) - broadcast bias to all ctx tokens
        ctx = ctx.to(self.dtype)
        
        if squeeze_output:
            ctx = ctx.squeeze(0)
        
        return ctx


class VisualEncoderWithContext(nn.Module):
    """
    Wrapper around CLIP ViT that inserts visual context tokens before CLS.
    Sequence: [C1] [C2] [C3] [CLS] [P1] [P2] ... [PN]
    Applies visual.proj at the end so output dim matches text (embed_dim);
    both image and text encoders use projection heads (visual.proj, text_projection).
    """
    def __init__(self, visual_encoder, visual_ctx_learner):
        super().__init__()
        self.visual = visual_encoder
        self.visual_ctx_learner = visual_ctx_learner
        self.n_ctx_vis = visual_ctx_learner.n_ctx_vis
        
    def forward(self, x, text_features=None):
        """
        Forward with visual context tokens inserted.
        
        Args:
            x: (B, C, H, W) input images
            text_features: (B, dim) optional text features for conditional visual ctx
        
        Returns:
            image_features: (B, dim) CLS token output
        """
        B = x.shape[0]
        visual = self.visual
        
        # Get visual context tokens (with conditional bias if text_features provided)
        visual_ctx = self.visual_ctx_learner(text_features)  # (B, n_ctx_vis, vis_dim)
        if visual_ctx.dim() == 2:
            visual_ctx = visual_ctx.unsqueeze(0).expand(B, -1, -1)
        
        # Get patch embeddings using ViT's conv1 (patch embedding)
        # For ViT-B/16: conv1 is Conv2d that patches the image
        x = visual.conv1(x)  # (B, embed_dim, grid_h, grid_w)
        grid_h, grid_w = x.shape[2], x.shape[3]
        num_patches = grid_h * grid_w
        x = x.reshape(B, x.shape[1], -1)  # (B, embed_dim, num_patches)
        x = x.permute(0, 2, 1)  # (B, num_patches, embed_dim)
        
        # Get CLS token (class_embedding may be 1D (embed_dim,) or 2D (1, embed_dim))
        cls_emb = visual.class_embedding
        if cls_emb.dim() == 1:
            cls_emb = cls_emb.unsqueeze(0)  # (1, embed_dim)
        cls_token = cls_emb.unsqueeze(0).expand(B, -1, -1)  # (B, 1, embed_dim)
        
        # Concatenate: [visual_ctx] [CLS] [patches]
        x = torch.cat([visual_ctx, cls_token, x], dim=1)  # (B, n_ctx_vis + 1 + num_patches, embed_dim)
        
        # Add positional embeddings
        # Original ViT pos_emb: [CLS] [P1] [P2] ... [PN] -> shape (num_patches + 1, embed_dim)
        pos_emb = visual.positional_embedding  # (num_patches + 1, embed_dim)
        cls_pos = pos_emb[0:1]  # (1, embed_dim) - CLS positional
        patch_pos = pos_emb[1:]  # (num_patches, embed_dim) - patch positionals
        
        # Create new positional embeddings: [ctx_pos (zeros)] [CLS_pos] [patch_pos]
        ctx_pos = torch.zeros(self.n_ctx_vis, pos_emb.shape[1], dtype=pos_emb.dtype, device=pos_emb.device)
        new_pos_emb = torch.cat([ctx_pos, cls_pos, patch_pos], dim=0)  # (n_ctx_vis + 1 + num_patches, embed_dim)
        x = x + new_pos_emb.unsqueeze(0)
        
        # Run through transformer
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # (seq_len, B, embed_dim) for transformer
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # (B, seq_len, embed_dim)
        
        # Extract CLS token (at position n_ctx_vis, after visual ctx tokens)
        cls_idx = self.n_ctx_vis
        x = visual.ln_post(x[:, cls_idx, :])  # (B, width) e.g. 768 for ViT-L
        # Apply visual projection head (same as original ViT forward) -> (B, output_dim) e.g. 512
        if getattr(visual, "proj", None) is not None:
            x = x @ visual.proj  # (B, output_dim) to match text_projection output
        return x


class ReverseMetaNet(nn.Module):
    """
    Opposite of CoCoOp meta_net: takes TEXT embedding and outputs a delta for IMAGE embedding.
    Used for 'image CoCoOp': text conditions the image (adapted_img = img_feats + reverse_meta_net(text_context)).
    Same 3-layer + LayerNorm structure as meta_net; input_dim = text/ctx_dim, output_dim = vis_dim.
    """
    def __init__(self, text_dim, image_dim):
        super().__init__()
        h1, h2 = text_dim // 4, text_dim // 16
        self.net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(text_dim, h1)),
            ("norm1", nn.LayerNorm(h1)),
            ("relu1", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(h1, h2)),
            ("norm2", nn.LayerNorm(h2)),
            ("relu2", nn.ReLU(inplace=True)),
            ("linear3", nn.Linear(h2, image_dim)),
        ]))

    def forward(self, text_features):
        """text_features: (B, dim) or (1, dim). Returns delta (same shape) to add to image features."""
        return self.net(text_features)


class ClipTestTimePromptTuning(nn.Module):
    def __init__(self, clip_model, normalization, arch_name, dataset_name, n_ctx=16,
                 ctx_init=None, class_token_pos='end', learned_cls=False, use_cocoop=False, use_reverse_cocoop=False):
        super(ClipTestTimePromptTuning, self).__init__()

        # setup the underlying CLIP model
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale.data
        self.normalize = normalization
        self.use_cocoop = use_cocoop
        self.use_reverse_cocoop = use_reverse_cocoop

        # get the class names form the dataset name
        class_names = get_class_names(dataset_name)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = ctx_dim  # ViT: same as image encoder output dim

        # prompt tuning - use CoCoOp if requested (image conditions text)
        if use_cocoop:
            self.prompt_learner = CoCoOpPromptLearner(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)
        else:
            self.prompt_learner = PromptLearner(clip_model, arch_name, class_names, n_ctx, ctx_init, class_token_pos, learned_cls)

        # reverse CoCoOp: text conditions image (text -> delta -> add to image embedding)
        if use_reverse_cocoop:
            self.reverse_meta_net = ReverseMetaNet(text_dim=ctx_dim, image_dim=vis_dim)
        else:
            self.reverse_meta_net = None

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_class_names(self, class_names):
        self.prompt_learner.reset_class_names(class_names)

    def get_text_features_zeroshot(self):
        """
        Simple text embedder: template "a photo of a {class}." with no learnable ctx.
        No CoOp/CoCoOp; used when use_reverse_cocoop=True.
        """
        pl = self.prompt_learner
        class_names = [n.replace("_", " ") for n in pl.class_names]
        prompts_str = ["a photo of a " + n + "." for n in class_names]
        tokenized_prompts = torch.cat([pl.tokenize(p) for p in prompts_str]).to(pl.ctx.device)
        prompts = pl.token_embedding(tokenized_prompts).type(self.dtype)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        return text_features

    def get_text_features(self, image_features=None):
        """
        Get text features from prompts.
        For CoCoOp, image_features are used to generate image-conditioned prompts.
        """
        if self.use_cocoop and image_features is not None:
            prompts = self.prompt_learner(image_features=image_features)  # (B, n_cls, n_tok, dim)
            B, n_cls, n_tok, dim = prompts.shape
            # Reshape to (B * n_cls, n_tok, dim) for text encoder
            prompts = prompts.view(B * n_cls, n_tok, dim)
            # Expand tokenized prompts for batch
            tokenized_prompts = self.prompt_learner.tokenized_prompts.unsqueeze(0).expand(B, -1, -1)
            tokenized_prompts = tokenized_prompts.reshape(B * n_cls, -1)
        else:
            prompts = self.prompt_learner()
            tokenized_prompts = self.prompt_learner.tokenized_prompts
        
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        if self.use_cocoop and image_features is not None:
            # Reshape back to (B, n_cls, dim)
            text_features = text_features.view(B, n_cls, -1)
            # Normalize (eps to avoid division by zero / NaN)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            # Normalize (eps to avoid division by zero / NaN)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        return text_features

    def forward(self, image, return_features=False):
        image = self.normalize(image.type(self.dtype))
        img_pre_features = self.image_encoder(image)
        # Normalize with eps to avoid division by zero / NaN
        image_features = img_pre_features / (img_pre_features.norm(dim=-1, keepdim=True) + 1e-8)

        if self.use_cocoop and self.use_reverse_cocoop:
            # Both: text CoCoOp (image conditions text) + image CoCoOp (text conditions image), trained simultaneously
            B = image_features.shape[0]
            text_features = self.get_text_features(image_features=image_features)  # (B, n_cls, dim)
            text_context = text_features.mean(dim=(0, 1), keepdim=True).squeeze(0)  # (1, dim)
            delta = self.reverse_meta_net(text_context.to(self.reverse_meta_net.net[0].weight.dtype))  # (1, dim)
            delta = delta.expand(B, -1).to(image_features.dtype)  # (B, dim), match image dtype
            adapted_image_features = image_features + delta  # (B, dim)
            text_features_flat = text_features.mean(dim=0)  # (n_cls, dim)
            # Ensure same dtype for matmul (fp16 vs fp32 mismatch)
            ad = adapted_image_features.to(self.dtype)
            tf = text_features_flat.to(self.dtype)
            logits = self.logit_scale.exp().clamp(max=100.0) * (ad @ tf.t())  # (B, n_cls)
            text_pre_features = text_features_flat
            image_features_out = adapted_image_features
            img_pre_out = img_pre_features
        elif self.use_reverse_cocoop:
            # Simple text embedder (zero-shot template) + reverse_meta_net-tuned image; no CoCoOp
            text_features = self.get_text_features_zeroshot()  # (n_cls, dim), "a photo of a {class}."
            text_context = text_features.mean(dim=0, keepdim=True)  # (1, dim)
            delta = self.reverse_meta_net(text_context.to(self.reverse_meta_net.net[0].weight.dtype))  # (1, dim)
            delta = delta.to(image_features.dtype)
            adapted_image_features = image_features + delta  # (B, dim)
            ad = adapted_image_features.to(self.dtype)
            tf = text_features.to(self.dtype)
            logits = self.logit_scale.exp().clamp(max=100.0) * (ad @ tf.t())
            text_features_flat = text_features
            text_pre_features = text_features
            image_features_out = adapted_image_features
            img_pre_out = img_pre_features  # pre-norm not modified for loss compatibility
        elif self.use_cocoop:
            # CoCoOp: generate image-conditioned prompts
            text_features = self.get_text_features(image_features=image_features)  # (B, n_cls, dim)
            # Compute logits per image
            B = image_features.shape[0]
            logits_list = []
            for i in range(B):
                img_feat_i = image_features[i:i+1]  # (1, dim)
                text_feat_i = text_features[i]  # (n_cls, dim)
                text_feat_i_norm = text_feat_i / (text_feat_i.norm(dim=-1, keepdim=True) + 1e-8)
                logit_i = self.logit_scale.exp().clamp(max=100.0) * img_feat_i @ text_feat_i_norm.t()  # (1, n_cls)
                logits_list.append(logit_i)
            logits = torch.cat(logits_list, dim=0)  # (B, n_cls)
            # For BATCLIP losses: average text features over batch to get (n_cls, dim)
            text_features_flat = text_features.mean(dim=0)  # (n_cls, dim)
            text_pre_features = text_features.mean(dim=0)
            image_features_out = image_features
            img_pre_out = img_pre_features
        else:
            # Standard TPT: static prompts
            text_features = self.get_text_features()
            logits = self.logit_scale.exp() * image_features @ text_features.t()
            text_features_flat = text_features
            text_pre_features = text_features
            image_features_out = image_features
            img_pre_out = img_pre_features

        if return_features:
            return logits, image_features_out, text_features_flat, img_pre_out, text_pre_features
        else:
            return logits


class ClipBMPET(nn.Module):
    """
    BMPETCLIP: BiModel Prompt and Embedding space TTA.
    - Text side: CoOp-style learned prompts + CoCoOp meta-net for image-conditioned prompt bias.
    - Image side: Visual context tokens [C1] [C2] [C3] inserted before CLS, with CoCoOp-style conditional bias from text.
    - Fusion: (img_enc, text_enc) -> 1024 -> 512 -> 128 (BN between layers), then a head to produce a prompt-space bias.
    - BMPET bias is added to the CoCoOp prompt context tokens (prompt space) and re-encoded by CLIP's text encoder.
    - Final pass: adapted_image @ adapted_text for logits, with image side influenced by visual context tokens.
    BNs in fusion, head, visual ctx learner, and CLIP are adapted during TTA.
    """
    def __init__(self, clip_model, normalization, arch_name, dataset_name, n_ctx=16, n_ctx_vis=4,
                 ctx_init=None, class_token_pos='end', learned_cls=False):
        super().__init__()
        self.base = ClipTestTimePromptTuning(
            clip_model, normalization, arch_name, dataset_name,
            n_ctx=n_ctx, ctx_init=ctx_init, class_token_pos=class_token_pos, learned_cls=learned_cls,
            use_cocoop=True, use_reverse_cocoop=False,
        )
        # Projected output dim (both image and text encoders output this after proj / text_projection)
        proj = getattr(clip_model.visual, "proj", None)
        embed_dim = proj.shape[1] if proj is not None else clip_model.ln_final.weight.shape[0]
        
        # Visual context prompt learner (CoCoOp-style for image side)
        self.visual_ctx_learner = VisualContextPromptLearner(clip_model, n_ctx_vis=n_ctx_vis, ctx_init=None)
        
        # Wrap image encoder to insert visual context tokens (outputs embed_dim after visual.proj)
        self.image_encoder = VisualEncoderWithContext(clip_model.visual, self.visual_ctx_learner)
        
        # Fusion: concat(image_features, text_mean) -> both are embed_dim after projection -> 2 * embed_dim
        # Keep fusion and head in float32 to avoid mat1/mat2 dtype mismatch (CLIP base may be fp16)
        fusion_input_dim = embed_dim * 2
        self.fusion = FusionMLP(input_dim=fusion_input_dim, hidden_dim=512, output_dim=128).float()
        self.head_text_bias = BiasHead(input_dim=128, hidden_dim=64, output_dim=embed_dim).float()

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def prompt_learner(self):
        return self.base.prompt_learner

    @property
    def normalize(self):
        return self.base.normalize

    def reset(self):
        self.base.reset()

    def reset_class_names(self, class_names):
        self.base.reset_class_names(class_names)

    def get_text_features(self, image_features=None):
        return self.base.get_text_features(image_features=image_features)

    def forward(self, image, return_features=False):
        # ----- First pass: base CLIP encoders (no visual ctx) -----
        image = self.base.normalize(image.type(self.dtype))
        B = image.shape[0]
        
        # Get initial image features (without visual ctx) to compute text features for conditioning
        # This is needed because visual ctx tokens are conditioned on text, but text is conditioned on image
        # So we do: image -> text (CoCoOp) -> visual ctx (conditioned on text) -> final image (with visual ctx)
        img_pre_features_no_ctx = self.base.image_encoder(image)  # (B, dim) - standard ViT
        image_features_no_ctx = img_pre_features_no_ctx / (img_pre_features_no_ctx.norm(dim=-1, keepdim=True) + 1e-8)
        
        # ----- CoCoOp text features (image-conditioned prompts, no BMPET bias yet) -----
        # These are used to (1) condition visual context tokens and (2) feed the fusion MLP.
        text_features_init = self.base.get_text_features(image_features=image_features_no_ctx)  # (B, n_cls, dim)
        text_mean = text_features_init.mean(dim=1)  # (B, dim)
        
        # ----- Image encoder WITH visual context tokens (conditioned on text_mean) -----
        img_pre_features = self.image_encoder(image, text_features=text_mean)  # (B, dim) - with visual ctx tokens
        image_features = img_pre_features / (img_pre_features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # ----- Fusion + text bias head: embeddings -> prompt-space bias (CoCoOp-style) -----
        # Fusion/heads may be float32; CLIP may be Half -> cast input to fusion dtype to avoid mat1/mat2 mismatch
        joint = torch.cat([image_features, text_mean], dim=-1)  # (B, 2 * dim)
        fusion_dtype = next(self.fusion.parameters()).dtype
        joint_f = joint.to(fusion_dtype)
        z = self.fusion(joint_f)  # (B, 128)
        bias_text = self.head_text_bias(z)  # z same dtype as fusion output
        bias_text = bias_text.to(image_features.dtype)
        
        # ----- Second pass (final): apply BMPET bias in prompt space and re-encode text -----
        # Rebuild CoCoOp prompts, then add a shared bias (π_BMPET) to all tokens,
        # similar to CoCoOp's v_m(x) = v_m + π but at the prompt level.
        pl = self.prompt_learner
        prompts = pl(image_features=image_features_no_ctx)  # (B, n_cls, n_tok, dim) CoCoOp prompts
        Bp, n_cls, n_tok, dim = prompts.shape
        assert Bp == B, "Batch size mismatch between prompts and image features in ClipBMPET."
        
        # Broadcast bias_text over all classes and tokens: (B, 1, 1, dim)
        bias_prompt = bias_text.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, dim)
        prompts_bmpet = prompts + bias_prompt.to(prompts.dtype)  # (B, n_cls, n_tok, dim)
        
        # Flatten for text encoder: (B * n_cls, n_tok, dim)
        prompts_flat = prompts_bmpet.view(B * n_cls, n_tok, dim)
        
        # Tokenized prompts: reuse CoOp tokenization, expanded for batch
        tokenized_prompts = pl.tokenized_prompts.unsqueeze(0).expand(B, -1, -1)  # (B, n_cls, seq_len)
        tokenized_prompts = tokenized_prompts.reshape(B * n_cls, -1)
        
        # Encode biased prompts to get final adapted text features
        text_features = self.base.text_encoder(prompts_flat, tokenized_prompts)  # (B * n_cls, dim)
        text_features = text_features.view(B, n_cls, -1)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        adapted_text = text_features  # (B, n_cls, dim)
        
        # ----- Final logits from adapted embeddings only -----
        logits_list = []
        for i in range(B):
            ad_img = image_features[i:i + 1].to(self.dtype)
            ad_txt = adapted_text[i].to(self.dtype)
            ad_txt = ad_txt / (ad_txt.norm(dim=-1, keepdim=True) + 1e-8)
            logit_i = self.base.logit_scale.exp().clamp(max=100.0) * (ad_img @ ad_txt.t())
            logits_list.append(logit_i)
        logits = torch.cat(logits_list, dim=0)  # (B, n_cls)  <- only logits; loss must use this
        
        # For BATCLIP losses (TTA uses logits + these; loss is computed after this forward)
        text_features_flat = adapted_text.mean(dim=0)  # (n_cls, dim) - adapted
        image_features_out = image_features  # already has visual ctx tokens
        img_pre_out = img_pre_features  # with visual ctx tokens (for InterMean)
        text_pre_features = text_features_flat

        if return_features:
            return logits, image_features_out, text_features_flat, img_pre_out, text_pre_features
        return logits


class MaPLePromptLearner(PromptLearner):
    """
    MaPLe-style prompt learner with shallow + deep compound prompts.

    This extends the base PromptLearner by adding a stack of additional
    context tokens (\"deep\" prompts). All prompts (shallow + deep) are
    concatenated in the context region of the text sequence, and projected
    to the visual width for use as visual prompt tokens.
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
        learned_cls: bool = False,
    ):
        super().__init__(
            clip_model,
            arch_name,
            class_names,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            class_token_pos=class_token_pos,
            learned_cls=learned_cls,
        )

        assert prompt_depth >= 1, "MaPLe prompt_depth must be >= 1"
        self.prompt_depth = prompt_depth

        # Deep compound prompts in text space (same dim as ctx / ln_final)
        self.deep_ctx = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype, device=self.device))
                for _ in range(self.prompt_depth - 1)
            ]
        )
        for p in self.deep_ctx:
            nn.init.normal_(p, std=0.02)

        # Projection from text ctx_dim -> visual embedding dim
        vis_dim = clip_model.visual.conv1.weight.shape[0]
        self.proj_vis = nn.Linear(self.ctx_dim, vis_dim)

        logger.info(
            "Initialized MaPLePromptLearner with n_ctx=%d, depth=%d (total tokens=%d)",
            self.n_ctx,
            self.prompt_depth,
            self.n_ctx * self.prompt_depth,
        )

    def _get_full_ctx(self):
        """
        Concatenate shallow and deep context tokens in text space.
        Returns:
            ctx_all: (n_ctx_total, ctx_dim)
        """
        ctx_list = [self.ctx] + list(self.deep_ctx)
        ctx_all = torch.cat(ctx_list, dim=0)  # (n_ctx_total, ctx_dim)
        return ctx_all

    def forward(self, init=None):
        """
        Build text prompts with concatenated shallow + deep context tokens.
        """
        ctx_base = init if init is not None else self._get_full_ctx()

        if ctx_base.dim() == 2:
            ctx = ctx_base.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx_base.size()[0] == self.n_cls:
            ctx = ctx_base.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        else:
            ctx = ctx_base

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.learned_cls:
            assert self.class_token_position == "end"

        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,
                        ctx,
                        cls,
                        suffix,
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,
                        ctx,
                        suffix,
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            if self.split_idx is not None:
                half_n_ctx = self.split_idx
            else:
                half_n_ctx = ctx.shape[1] // 2
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)
        elif self.class_token_position == "front":
            prompts_list = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts_list.append(prompt)
            prompts = torch.cat(prompts_list, dim=0)
        else:
            raise ValueError(
                f"Class token position '{self.class_token_position}' is not supported."
                f" Choose from: end, middle, front"
            )

        return prompts

    def get_visual_prompts(self):
        """
        Project shallow + deep text prompts into visual embedding space.
        Returns:
            visual_ctx: (n_ctx_total, vis_dim)
        """
        ctx_all = self._get_full_ctx()  # (n_ctx_total, ctx_dim)
        visual_ctx = self.proj_vis(ctx_all.to(self.proj_vis.weight.dtype))
        return visual_ctx


class ClipMaPLe(nn.Module):
    """
    MaPLe-style CLIP wrapper for test-time adaptation.

    - Uses MaPLePromptLearner to build multimodal prompts.
    - Adds prompt tokens on the text side via TextEncoder.
    - Adds prompt tokens on the image side as extra tokens before CLS, similar
      to VisualEncoderWithContext.
    - Exposes a TTA-friendly API with return_features=True returning the
      components needed by BATCLIP losses.
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

        self.prompt_learner = MaPLePromptLearner(
            clip_model,
            arch_name,
            class_names,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            prompt_depth=prompt_depth,
            class_token_pos=class_token_pos,
            learned_cls=False,
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.visual.conv1.weight.dtype

    @property
    def prompt_dim(self):
        return self.prompt_learner.ctx_dim

    def _encode_image_with_prompts(self, image, visual_ctx_tokens):
        """
        Encode image with MaPLe visual prompt tokens inserted before CLS.
        This closely follows VisualEncoderWithContext but uses static prompts.
        """
        B = image.shape[0]
        visual = self.image_encoder

        # visual conv1: patch embedding
        x = visual.conv1(image)  # (B, embed_dim, grid_h, grid_w)
        grid_h, grid_w = x.shape[2], x.shape[3]
        num_patches = grid_h * grid_w
        x = x.reshape(B, x.shape[1], -1)  # (B, embed_dim, num_patches)
        x = x.permute(0, 2, 1)  # (B, num_patches, embed_dim)

        # CLS token
        cls_emb = visual.class_embedding
        if cls_emb.dim() == 1:
            cls_emb = cls_emb.unsqueeze(0)
        cls_token = cls_emb.unsqueeze(0).expand(B, -1, -1)  # (B, 1, embed_dim)

        # Visual prompt tokens
        if visual_ctx_tokens.dim() == 2:
            visual_ctx = visual_ctx_tokens.unsqueeze(0).expand(B, -1, -1)
        else:
            visual_ctx = visual_ctx_tokens

        # Concatenate: [visual_ctx] [CLS] [patches]
        x = torch.cat([visual_ctx, cls_token, x], dim=1)

        # Positional embeddings
        pos_emb = visual.positional_embedding  # (num_patches + 1, dim)
        cls_pos = pos_emb[0:1]
        patch_pos = pos_emb[1:]

        n_ctx_vis = visual_ctx.shape[1]
        ctx_pos = torch.zeros(
            n_ctx_vis,
            pos_emb.shape[1],
            dtype=pos_emb.dtype,
            device=pos_emb.device,
        )
        new_pos_emb = torch.cat([ctx_pos, cls_pos, patch_pos], dim=0)
        x = x + new_pos_emb.unsqueeze(0)

        # Transformer
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        # CLS at index n_ctx_vis
        cls_idx = n_ctx_vis
        x = visual.ln_post(x[:, cls_idx, :])
        if getattr(visual, "proj", None) is not None:
            x = x @ visual.proj
        return x

    def forward(self, image, return_features: bool = False):
        # Normalize input to CLIP dtype
        image = self.normalize(image.type(self.dtype))

        # ----- Text side: MaPLe prompts -----
        prompts = self.prompt_learner()  # (n_cls, n_tok, ctx_dim)
        tokenized_prompts = self.tokenized_prompts

        # Text pre-features and normalized features
        text_pre_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_pre_features / (
            text_pre_features.norm(dim=-1, keepdim=True) + 1e-8
        )  # (n_cls, dim)

        # ----- Image side: visual prompts -----
        visual_ctx_tokens = self.prompt_learner.get_visual_prompts()  # (n_ctx_total, vis_dim)
        img_pre_features = self._encode_image_with_prompts(
            image, visual_ctx_tokens
        )  # (B, dim)
        image_features = img_pre_features / (
            img_pre_features.norm(dim=-1, keepdim=True) + 1e-8
        )

        # ----- Logits -----
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (image_features @ text_features.t())

        if not return_features:
            return logits

        # BATCLIP-compatible outputs
        image_features_out = image_features  # normalized
        text_features_flat = text_features  # (n_cls, dim), normalized

        return logits, image_features_out, text_features_flat, img_pre_features, text_pre_features
