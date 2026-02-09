"""
MaPLeBATCLIP: Test-time adaptation for MaPLe-style CLIP using BATCLIP losses.

- Adapts MaPLe deep compound prompts and LayerNorm layers in the encoders.
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
      * MaPLe prompt learner deep compound prompts (MaPLePromptLearner.deep_ctx).
      * All LayerNorm (nn.LayerNorm) layers inside image_encoder and text_encoder.
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
        - MaPLe prompt learner deep compound prompts (deep_ctx).
        - All nn.LayerNorm layers in image_encoder and text_encoder.
        """
        self.model.eval()
        self.model.requires_grad_(False)

        adapted_names = []

        # 1) Enable MaPLe deep prompts
        prompt_learner = getattr(self.model, "prompt_learner", None)
        if prompt_learner is not None:
            for name, p in prompt_learner.named_parameters():
                # Only deep compound prompts
                if "deep_ctx" in name:
                    p.requires_grad_(True)
                    adapted_names.append(f"prompt_learner.{name}")

        # 2) Enable all LayerNorms in image and text encoders
        for name, m in self.model.named_modules():
            if not isinstance(m, nn.LayerNorm):
                continue

            in_image = "image_encoder" in name
            in_text = "text_encoder" in name

            if in_image or in_text:
                m.requires_grad_(True)
                adapted_names.append(name)

        if adapted_names:
            logger.info("[MaPLeBATCLIP] Adapted parameters/layers (%d):", len(adapted_names))
            for nm in sorted(adapted_names):
                logger.info("  - %s", nm)

    def collect_params(self):
        """
        Collect trainable parameters:
        - LayerNorm weights/biases in encoders.
        - MaPLe deep prompt parameters (deep_ctx).
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ("weight", "bias") and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")

        # MaPLe deep prompts
        prompt_learner = getattr(self.model, "prompt_learner", None)
        if prompt_learner is not None:
            for np, p in prompt_learner.named_parameters():
                if "deep_ctx" in np and p.requires_grad:
                    params.append(p)
                    names.append(f"prompt_learner.{np}")

        return params, names

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        Single forward with BATCLIP losses:
        - Uses logits + image/text pre-features from MaPLe CLIP wrapper.
        """
        imgs_test = x[0]

        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(imgs_test, return_features=True)
        else:
            outputs = self.model(imgs_test, return_features=True)

        logits, image_features, text_features_flat, img_pre_features, text_pre_features = outputs

        # Compute BATCLIP losses
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

