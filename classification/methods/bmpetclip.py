"""
BMPETCLIP: BiModel Prompt and Embedding space TTA.
- Text side: CoOp + CoCoOp (image-conditioned text prompts).
- Image side: visual context tokens (text-conditioned image prompts) inserted before CLS.
- Fusion: (img_emb, text_emb) -> 128, then a head to bias text embeddings.
- During TTA, we adapt:
  - Norm layers in CLIP visual/text encoders,
  - Norm layers in fusion + text-bias head,
  - LayerNorms in the visual context meta-net.
  using BATCLIP losses (Entropy, I2T, InterMean) + optional TPT avg-entropy.
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
class BMPETCLIP(TTAMethod):
    """
    BMPETCLIP: Test-time adaptation with:
    - CoOp + CoCoOp prompts and visual context tokens, with an extra BMPET prompt-space bias learned via fusion.
    - Norm layers in CLIP visual/text encoders, fusion, text-bias head, and visual context meta-net adapted at TTA.
    - BATCLIP losses: Entropy, I2T, InterMean (+ optional TPT avg-entropy) drive the online adaptation.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.entropy_loss = Entropy()
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()

        self.selection_p = cfg.TPT.SELECTION_P if hasattr(cfg.TPT, "SELECTION_P") else 0.1
        self.lambda_ent = cfg.TPT.LAMBDA_ENT if hasattr(cfg.TPT, "LAMBDA_ENT") else 0.0
        self.unimodal_image_only = (
            cfg.MODEL.UNIMODAL_IMAGE_ONLY if hasattr(cfg.MODEL, "UNIMODAL_IMAGE_ONLY") else False
        )

        self.scaler = torch.cuda.amp.GradScaler() if cfg.MIXED_PRECISION else None

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        Single forward = first pass (encoders) -> fusion+heads -> edit -> second pass (logits).
        Loss is computed only on the final-pass logits (after 2 passes).
        """
        imgs_test = x[0]

        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(imgs_test, return_features=True)
        else:
            outputs = self.model(imgs_test, return_features=True)

        # logits = second-pass (adapted_image @ adapted_text); loss uses these only
        logits, image_features, text_features_flat, img_pre_features, text_pre_features = outputs

        if self.scaler:
            with torch.cuda.amp.autocast():
                loss = self.entropy_loss(logits).mean(0)
                if not self.unimodal_image_only:
                    loss = loss - self.i2t_loss(logits, img_pre_features, text_features_flat)
                    loss = loss - self.inter_mean_loss(logits, img_pre_features)
                if self.lambda_ent > 0 and self.selection_p > 0:
                    logits_conf, _ = select_confident_samples(logits, self.selection_p)
                    loss = loss + self.lambda_ent * avg_entropy(logits_conf)
        else:
            loss = self.entropy_loss(logits).mean(0)
            if not self.unimodal_image_only:
                loss = loss - self.i2t_loss(logits, img_pre_features, text_features_flat)
                loss = loss - self.inter_mean_loss(logits, img_pre_features)
            if self.lambda_ent > 0 and self.selection_p > 0:
                logits_conf, _ = select_confident_samples(logits, self.selection_p)
                loss = loss + self.lambda_ent * avg_entropy(logits_conf)

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

    def configure_model(self):
        """
        Adapt norm layers in:
        - CLIP visual encoder (base.image_encoder.*),
        - CLIP text encoder (base.text_encoder.*),
        - BMPET fusion MLP (fusion.*),
        - BMPET text-bias head (head_text_bias.*),
        - Visual context meta-net LayerNorms (visual_ctx_learner.meta_net.norm1/norm2).
        """
        self.model.eval()
        self.model.requires_grad_(False)

        adapted_layer_names = []
        for name, m in self.model.named_modules():
            if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                continue

            in_fusion_head = "fusion." in name or "head_text_bias." in name
            in_visual_ctx = "visual_ctx_learner.meta_net." in name and (".norm1" in name or ".norm2" in name)
            in_clip = "base.image_encoder." in name or "base.text_encoder." in name

            if in_fusion_head or in_visual_ctx or in_clip:
                m.requires_grad_(True)
                adapted_layer_names.append(name)
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
                elif isinstance(m, nn.BatchNorm1d):
                    # BN1d needs train mode for running stats
                    m.train()

        if adapted_layer_names:
            logger.info("[BMPETCLIP] Layers adapted at TTA (%d total):", len(adapted_layer_names))
            for nm in sorted(adapted_layer_names):
                logger.info("  - %s", nm)

    def collect_params(self):
        """
        Collect trainable parameters from:
        - Norm layers in fusion + text-bias head,
        - LayerNorms in visual_ctx_learner.meta_net (norm1/norm2),
        - Norm layers in CLIP visual/text encoders.
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                continue
            if ("fusion." in nm or "head_text_bias." in nm or 
                ("visual_ctx_learner.meta_net." in nm and (".norm1" in nm or ".norm2" in nm)) or
                "base.image_encoder." in nm or "base.text_encoder." in nm):
                for np, p in m.named_parameters():
                    if np in ("weight", "bias"):
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
