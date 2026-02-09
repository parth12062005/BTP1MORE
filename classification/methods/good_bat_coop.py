"""
GoodBATCoop: Two-stage test-time adaptation for CoCoOp-CLIP.
1. Adapt image and text encoder with BATCLIP losses (entropy, I2T, InterMean).
2. Adapt meta_net last layer (linear3) with TPT loss (avg entropy on confident samples).
"""

import torch
import torch.nn as nn
from copy import deepcopy
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, I2TLoss, InterMeanLoss
from methods.tpt import select_confident_samples, avg_entropy


@ADAPTATION_REGISTRY.register()
class GoodBATCoop(TTAMethod):
    """
    GoodBATCoop: Two-stage adaptation:
    - Stage 1: Update image + text encoder (norm layers) with BATCLIP losses.
    - Stage 2: Update prompt_learner.meta_net.linear2 with TPT-style avg-entropy loss.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.entropy_loss = Entropy()
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()

        self.selection_p = cfg.TPT.SELECTION_P if hasattr(cfg.TPT, "SELECTION_P") else 0.1
        self.unimodal_image_only = (
            cfg.MODEL.UNIMODAL_IMAGE_ONLY if hasattr(cfg.MODEL, "UNIMODAL_IMAGE_ONLY") else False
        )

        self.scaler = torch.cuda.amp.GradScaler() if cfg.MIXED_PRECISION else None

        # Replace single optimizer with two: encoder (BATCLIP) and meta last layer (TPT)
        self._encoder_params = getattr(self, "_encoder_params", [])
        self._meta_params = getattr(self, "_meta_params", [])
        self.optimizer_encoder = (
            self._setup_optimizer_for(self._encoder_params) if len(self._encoder_params) > 0 else None
        )
        self.optimizer_meta = (
            self._setup_optimizer_for(self._meta_params) if len(self._meta_params) > 0 else None
        )
        # Keep self.optimizer for base API; copy/load use both optimizers
        self.optimizer = self.optimizer_encoder if self.optimizer_encoder is not None else self.optimizer_meta
        # Re-save state so reset() restores both optimizers (base saved (None, None) before they existed)
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def _setup_optimizer_for(self, params):
        if len(params) == 0:
            return None
        cfg = self.cfg
        if cfg.OPTIM.METHOD == "Adam":
            return torch.optim.Adam(
                params, lr=cfg.OPTIM.LR, betas=(cfg.OPTIM.BETA, 0.999), weight_decay=cfg.OPTIM.WD
            )
        elif cfg.OPTIM.METHOD == "AdamW":
            return torch.optim.AdamW(
                params, lr=cfg.OPTIM.LR, betas=(cfg.OPTIM.BETA, 0.999), weight_decay=cfg.OPTIM.WD
            )
        elif cfg.OPTIM.METHOD == "SGD":
            return torch.optim.SGD(
                params,
                lr=cfg.OPTIM.LR,
                momentum=cfg.OPTIM.MOMENTUM,
                dampening=cfg.OPTIM.DAMPENING,
                weight_decay=cfg.OPTIM.WD,
                nesterov=cfg.OPTIM.NESTEROV,
            )
        return None

    def copy_model_and_optimizer(self):
        model_states = [deepcopy(model.state_dict()) for model in self.models]
        enc_state = (
            deepcopy(self.optimizer_encoder.state_dict())
            if getattr(self, "optimizer_encoder", None) is not None
            else None
        )
        meta_state = (
            deepcopy(self.optimizer_meta.state_dict())
            if getattr(self, "optimizer_meta", None) is not None
            else None
        )
        optimizer_state = (enc_state, meta_state)
        return model_states, optimizer_state

    def load_model_and_optimizer(self):
        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)
        enc_state, meta_state = self.optimizer_state
        if enc_state is not None and self.optimizer_encoder is not None:
            self.optimizer_encoder.load_state_dict(enc_state)
        if meta_state is not None and self.optimizer_meta is not None:
            self.optimizer_meta.load_state_dict(meta_state)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        def forward():
            if self.scaler:
                with torch.cuda.amp.autocast():
                    return self.model(imgs_test, return_features=True)
            return self.model(imgs_test, return_features=True)

        # ----- Stage 1: Adapt encoder with BATCLIP losses -----
        outputs = forward()
        logits, image_features, text_features_flat, img_pre_features, text_pre_features = outputs

        if self.scaler:
            with torch.cuda.amp.autocast():
                loss_bat = self.entropy_loss(logits).mean(0)
                if not self.unimodal_image_only:
                    loss_bat = loss_bat - self.i2t_loss(logits, img_pre_features, text_features_flat)
                    loss_bat = loss_bat - self.inter_mean_loss(logits, img_pre_features)
        else:
            loss_bat = self.entropy_loss(logits).mean(0)
            if not self.unimodal_image_only:
                loss_bat = loss_bat - self.i2t_loss(logits, img_pre_features, text_features_flat)
                loss_bat = loss_bat - self.inter_mean_loss(logits, img_pre_features)

        if self.optimizer_encoder is not None:
            self.optimizer_encoder.zero_grad()
            if self.scaler:
                self.scaler.scale(loss_bat).backward()
                self.scaler.step(self.optimizer_encoder)
                self.scaler.update()
            else:
                loss_bat.backward()
                self.optimizer_encoder.step()

        # ----- Stage 2: Adapt meta_net.linear3 with TPT loss -----
        if self.optimizer_meta is not None:
            outputs = forward()
            logits = outputs[0]
            logits_conf, _ = select_confident_samples(logits, self.selection_p)
            loss_tpt = avg_entropy(logits_conf)

            self.optimizer_meta.zero_grad()
            if self.scaler:
                self.scaler.scale(loss_tpt).backward()
                self.scaler.step(self.optimizer_meta)
                self.scaler.update()
            else:
                loss_tpt.backward()
                self.optimizer_meta.step()

        return logits.detach()

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        # Encoder: enable norm layers only in image_encoder and text_encoder
        for nm, m in self.model.named_modules():
            if ("image_encoder" in nm or "text_encoder" in nm) and isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):
                m.train()
                m.requires_grad_(True)
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

        # Prompt learner: only meta_net.linear3 (last layer)
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name and "meta_net.linear3" in name:
                param.requires_grad_(True)

    def collect_params(self):
        encoder_params = []
        encoder_names = []
        meta_params = []
        meta_names = []

        for nm, m in self.model.named_modules():
            if ("image_encoder" in nm or "text_encoder" in nm) and isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):
                for np, p in m.named_parameters():
                    if np in ("weight", "bias") and p.requires_grad:
                        encoder_params.append(p)
                        encoder_names.append(f"{nm}.{np}")

        for name, param in self.model.named_parameters():
            if "prompt_learner" in name and "meta_net.linear3" in name and param.requires_grad:
                meta_params.append(param)
                meta_names.append(name)

        self._encoder_params = encoder_params
        self._meta_params = meta_params
        all_params = encoder_params + meta_params
        all_names = encoder_names + meta_names
        return all_params, all_names
