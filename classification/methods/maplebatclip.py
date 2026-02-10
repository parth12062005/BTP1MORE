"""
MaPLeBATCLIP: Test-time adaptation for MaPLe-style CLIP using BATCLIP losses.

- Adapts deep prompts (compound_prompts_text), bias of compound_prompt_projections,
  and LayerNorm layers in the encoders.
- Uses Entropy + I2T + InterMean (+ optional TPT avg-entropy) as in BMPETCLIP.
"""

import logging
import torch
import torch.nn as nn

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, I2TLoss, InterMeanLoss
from methods.tpt import select_confident_samples, avg_entropy


logger = logging.getLogger(__name__)


@ADAPTATION_REGISTRY.register()
class MaPLeBATCLIP(TTAMethod):
    """
    MaPLeBATCLIP: Test-time adaptation for MaPLe-style CLIP.

    - Target parameters:
      * Deep prompts (compound_prompts_text).
      * Bias of compound_prompt_projections (visual projection layers).
      * All LayerNorm/BatchNorm/GroupNorm layers inside encoders.
    - Loss:
      * BATCLIP-style: Entropy + I2T + InterMean (+ optional TPT avg-entropy).
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # Losses
        self.entropy_loss = Entropy()
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()

        # TPT-style options
        self.selection_p = cfg.TPT.SELECTION_P if hasattr(cfg.TPT, "SELECTION_P") else 0.1
        self.lambda_ent = cfg.TPT.LAMBDA_ENT if hasattr(cfg.TPT, "LAMBDA_ENT") else 0.0
        self.unimodal_image_only = (
            cfg.MODEL.UNIMODAL_IMAGE_ONLY if hasattr(cfg.MODEL, "UNIMODAL_IMAGE_ONLY") else False
        )

        # Mixed precision scaler (overrides base.scaler if needed)
        self.scaler = torch.cuda.amp.GradScaler() if cfg.MIXED_PRECISION else None

    def configure_model(self):
        """
        Enable gradients only for:
        - Deep prompts (compound_prompts_text).
        - Bias of compound_prompt_projections (visual projection layers, not weights).
        - All nn.LayerNorm layers in image_encoder and text_encoder.
        """
        self.model.eval()
        self.model.requires_grad_(False)

        adapted_names = []

        # 1) Enable deep prompts and bias of compound_prompt_projections
        prompt_learner = getattr(self.model, "prompt_learner", None)
        if prompt_learner is not None:
            for name, p in prompt_learner.named_parameters():
                if "compound_prompts_text" in name:
                    p.requires_grad_(True)
                    adapted_names.append(f"prompt_learner.{name}")
                elif "compound_prompt_projections" in name and "bias" in name:
                    p.requires_grad_(True)
                    adapted_names.append(f"prompt_learner.{name}")

        # 2) Enable all LayerNorms,batchnorm1d,batchnorm2d,groupnorm in image and text encoders
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, nn.BatchNorm2d):
                m.train()
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def collect_params(self):
        """
        Collect trainable parameters:
        - LayerNorm weights/biases in encoders.
        - Deep prompts (compound_prompts_text).
        - Bias of compound_prompt_projections.
        """
        params = []
        names = []

        # LayerNorms,batchnorm1d,batchnorm2d,groupnorm in encoders
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")

        # Deep prompts and bias of compound_prompt_projections
        prompt_learner = getattr(self.model, "prompt_learner", None)
        if prompt_learner is not None:
            for np, p in prompt_learner.named_parameters():
                if p.requires_grad and (
                    "compound_prompts_text" in np or
                    ("compound_prompt_projections" in np and "bias" in np)
                ):
                    params.append(p)
                    names.append(f"prompt_learner.{np}")

        return params, names

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        Single forward with BATCLIP losses:
        - Uses logits + image/text pre-features from MaPLe CLIP wrapper.
        Follows BMPETCLIP pattern.
        """
        imgs_test = x[0]

        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(imgs_test, return_features=True)
        else:
            outputs = self.model(imgs_test, return_features=True)

        logits, image_features, text_features_flat, img_pre_features, text_pre_features = outputs

        # Compute BATCLIP losses (same pattern as BMPETCLIP)
        if self.scaler:
            with torch.cuda.amp.autocast():
                loss = self._compute_loss(logits, img_pre_features, text_features_flat)
        else:
            loss = self._compute_loss(logits, img_pre_features, text_features_flat)

        # Optimizer step
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return logits.detach()

    def _compute_loss(self, logits, img_pre_features, text_features_flat):
        """
        BATCLIP-style loss:
        Entropy + I2T + InterMean (+ optional avg-entropy on confident samples).
        """
        # Main entropy loss
        loss = self.entropy_loss(logits).mean(0)

        # BATCLIP bimodal losses (unless in unimodal-image-only mode)
        if not self.unimodal_image_only:
            loss = loss - self.i2t_loss(logits, img_pre_features, text_features_flat)
            loss = loss - self.inter_mean_loss(logits, img_pre_features)
        else:
            loss = loss - self.inter_mean_loss(logits, img_pre_features)

        # Optional TPT-style avg-entropy on confident samples
        if self.lambda_ent > 0 and self.selection_p > 0:
            logits_conf, _ = select_confident_samples(logits, self.selection_p)
            loss = loss + self.lambda_ent * avg_entropy(logits_conf)

        return loss

