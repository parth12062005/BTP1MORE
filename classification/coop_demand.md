# CoCoOp-BATCLIP — Full File & Function Tree

This document lists every file and function used by **cocoop_batclip** and how they connect.

---

## 1. Entry point & config

| File | Role |
|------|------|
| `test_time.py` | Main entry: loads config, builds model, gets adaptation from registry, runs evaluation. |
| `cfgs/cifar10_c/cocoop_batclip.yaml` | Config for CoCoOp-BATCLIP (adaptation name, CLIP/TPT/OPTIM/TEST, etc.). |
| `conf.py` | Global `cfg` (YACS); merged with YAML. |
| `utils/registry.py` | `ADAPTATION_REGISTRY`: registers `CoCoOpBATCLIP`, lookup by name (e.g. `cocoopbatclip`). |

**Flow:**  
`test_time.py` → `load_cfg_from_args(description)` (reads YAML + conf) → `get_model(cfg, ...)` → `ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg, base_model, num_classes)` → `CoCoOpBATCLIP` instance.

---

## 2. Method: CoCoOp-BATCLIP

| File | Contents |
|------|----------|
| **`methods/cocoop_batclip.py`** | Defines `CoCoOpBATCLIP(TTAMethod)`: configures model, collects params, implements `forward_and_adapt` with entropy + BATCLIP losses + TPT avg-entropy. |
| **`methods/__init__.py`** | Imports `CoCoOpBATCLIP` and registers it; exposes it in `__all__`. |

### Functions/classes in `methods/cocoop_batclip.py`

| Name | Role |
|------|------|
| `CoCoOpBATCLIP.__init__` | Saves cfg/model/num_classes; creates `Entropy`, `I2TLoss`, `InterMeanLoss`; reads TPT/UNIMODAL options; optional `GradScaler`. |
| `CoCoOpBATCLIP.configure_model` | `model.eval()`, freeze all; then enable train+grad for (1) BN/LN/GN in full model, (2) `prompt_learner` params except `token_embedding`. |
| `CoCoOpBATCLIP.collect_params` | Collects: image encoder norm params, text encoder norm params, prompt_learner params (excluding `token_embedding`). |
| `CoCoOpBATCLIP.forward_and_adapt` | Forward with `return_features=True`; loss = entropy − I2TLoss − InterMeanLoss (+ optional TPT avg_entropy); backward; returns detached logits. |

---

## 3. Base TTA class

| File | Contents |
|------|----------|
| **`methods/base.py`** | Defines `TTAMethod(nn.Module)`: episodic/reset, sliding-window/single-sample, optimizer setup, reset/copy/load state. |

### Used by CoCoOp-BATCLIP from `methods/base.py`

| Name | Role |
|------|------|
| `TTAMethod.__init__` | cfg, model, num_classes; episodic/steps; calls `configure_model`, `collect_params`, `setup_optimizer`; copies model/optimizer state; mixed precision. |
| `TTAMethod.forward` | Dispatches batch vs single-sample; for batch: loop `forward_and_adapt` then return outputs. |
| `TTAMethod.setup_optimizer` | Builds Adam/AdamW/SGD from `cfg.OPTIM` (CoCoOp-BATCLIP uses AdamW). |
| `TTAMethod.copy_model_and_optimizer` | Deepcopy model and optimizer state for reset. |
| `TTAMethod.load_model_and_optimizer` | Restores from copied state (used on reset). |
| `TTAMethod.reset` | Calls `load_model_and_optimizer`. |

---

## 4. Losses (BATCLIP & entropy)

| File | Contents |
|------|----------|
| **`utils/losses.py`** | Defines `Entropy`, `I2TLoss`, `InterMeanLoss` (and other losses not used by cocoop_batclip). |

### Functions/classes used by CoCoOp-BATCLIP

| Name | Role |
|------|------|
| `Entropy.__call__(logits)` | \(-\sum_c p_c \log p_c\) per sample (softmax entropy). |
| `I2TLoss.__call__(logits, img_feats, text_norm_feats)` | Per-class mean image embedding dotted with text embedding; encourages image–text alignment. |
| `InterMeanLoss.__call__(logits, img_feats)` | Per-class mean image features, normalized; 1 − cosine_sim between different classes (push class means apart). |

---

## 5. TPT helpers (confident samples & avg entropy)

| File | Contents |
|------|----------|
| **`methods/tpt.py`** | Defines `select_confident_samples`, `avg_entropy`, and the `TPT` method class. |

### Functions used by CoCoOp-BATCLIP (not the TPT class)

| Name | Role |
|------|------|
| `select_confident_samples(logits, top)` | Sort by entropy ascending; keep top fraction `top` of samples; return their logits and indices. |
| `avg_entropy(outputs)` | Log-softmax over samples, then average; return negative sum of (exp * log) (avg predictive entropy). |

---

## 6. Model construction & forward

| File | Role |
|------|------|
| **`models/model.py`** | `get_model()`: when `USE_CLIP` and adaptation in `["tpt", "CoCoOpBATCLIP", "cocoopbatclip", "cocoop_batclip"]`, wraps CLIP in `ClipTestTimePromptTuning(..., use_cocoop=True)`. |
| **`models/custom_clip.py`** | `ClipTestTimePromptTuning`, `TextEncoder`, `PromptLearner`, `CoCoOpPromptLearner` (image-conditioned prompts). |

**Note:** `models/cocoop.py` (standalone CoCoOp CLIP wrapper) is **not** used by the cocoop_batclip pipeline; the code path uses only `custom_clip.py`.

### In `models/model.py`

| Name | Role |
|------|------|
| `get_model(cfg, num_classes, device)` | If CLIP + cocoop_batclip: `create_model_and_transforms` (open_clip) → wrap in `ClipTestTimePromptTuning(..., use_cocoop=True)`; optional CoOp checkpoint load for `ctx`. |

### In `models/custom_clip.py`

| Name | Role |
|------|------|
| `create_model_and_transforms` | From `open_clip`: load CLIP backbone and preprocess. |
| `get_class_names(dataset_name)` | From `datasets.cls_names`: return class names for the dataset. |
| `TextEncoder` | Wraps CLIP transformer + ln_final + text_projection; `forward(prompts, tokenized_prompts)` → text features. |
| `PromptLearner` | Base: ctx vectors, token_prefix/suffix, `reset_class_names`, `forward(init=None)` → prompts for all classes. |
| `CoCoOpPromptLearner(PromptLearner)` | Adds `meta_net(image_features)` to shift ctx per image; `forward(image_features=None, init=None)` → (B, n_cls, n_tok, dim) or (n_cls, n_tok, dim). |
| `ClipTestTimePromptTuning` | Holds image_encoder, text_encoder, normalize, prompt_learner (CoCoOp or not). `forward(image, return_features=False)` → logits and optionally (logits, img_feats, text_feats, img_pre_feats, text_pre_feats). |

### Forward path used by CoCoOp-BATCLIP

1. `CoCoOpBATCLIP.forward_and_adapt(x)` receives batch `x` (e.g. `[imgs_test]`).
2. Calls `self.model(imgs_test, return_features=True)`:
   - `ClipTestTimePromptTuning.forward(imgs_test, return_features=True)`:
     - Normalize → image_encoder → `img_pre_features`, normalized `image_features`.
     - `get_text_features(image_features)` → CoCoOp prompts per image → TextEncoder → `text_features` (B, n_cls, dim).
     - Logits per image: `logit_scale.exp() * image_features[i] @ text_features[i].T`.
   - Returns `(logits, image_features, text_features_flat, img_pre_features, text_pre_features)`.
3. Loss: `Entropy(logits).mean(0)` − `I2TLoss(logits, img_pre_features, text_features)` − `InterMeanLoss(logits, img_pre_features)`.
4. If `lambda_ent > 0` and `selection_p > 0`: `select_confident_samples(logits, selection_p)` → `avg_entropy(logits_conf)` added with weight `lambda_ent`.
5. Backward and optimizer step (with optional GradScaler).

---

## 7. Data & datasets

| File | Role |
|------|------|
| **`datasets/cls_names.py`** | `get_class_names(dataset_name)`: returns list of class names (e.g. cifar10_classes) for prompts. |
| **`datasets/data_loading.py`** | `get_test_loader(...)`: builds the test DataLoader (used in `test_time.py` for evaluation). |

Used indirectly: model and prompt learner need class names (via `get_class_names` in `get_model` / `ClipTestTimePromptTuning`); evaluation loop uses `get_test_loader`.

---

## 8. Utils used at runtime

| File | Role |
|------|------|
| **`utils/registry.py`** | `ADAPTATION_REGISTRY.register()` / `.get()` for resolving `CoCoOpBATCLIP`. |
| **`utils/eval_utils.py`** | Used by `test_time.py` for accuracy / domain evaluation (e.g. `get_accuracy`, `eval_domain_dict`). |
| **`utils/misc.py`** | Helpers (e.g. `print_memory_info`) if used in `test_time.py`. |
| **`utils/progress_tracker.py`** | `DomainProgressTracker` for logging progress. |
| **`utils/visualization.py`** | e.g. `create_results_visualization` for plots. |

---

## 9. Dependency tree (file-level)

```
test_time.py
├── conf.py                          # cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq
├── models/model.py                 # get_model
│   ├── open_clip (create_model_and_transforms)
│   ├── models/custom_clip.py       # ClipTestTimePromptTuning, TextEncoder, PromptLearner, CoCoOpPromptLearner
│   │   ├── open_clip (get_tokenizer)
│   │   └── datasets/cls_names.py   # get_class_names
│   └── (optional) torch.load for CoOp ctx
├── utils/registry.py               # ADAPTATION_REGISTRY
├── methods/__init__.py            # imports CoCoOpBATCLIP (registers it)
├── methods/cocoop_batclip.py      # CoCoOpBATCLIP
│   ├── methods/base.py            # TTAMethod
│   │   └── models/model.py        # ResNetDomainNet126 (only for copy_model branch, not CLIP path)
│   ├── utils/registry.py          # ADAPTATION_REGISTRY.register
│   ├── utils/losses.py            # Entropy, I2TLoss, InterMeanLoss
│   └── methods/tpt.py             # select_confident_samples, avg_entropy
├── datasets/data_loading.py       # get_test_loader
├── utils/eval_utils.py            # get_accuracy, eval_domain_dict
├── utils/progress_tracker.py      # DomainProgressTracker
└── utils/visualization.py         # create_results_visualization
```

---

## 10. Config (cocoop_batclip) → code mapping

| Config key | Where used |
|------------|------------|
| `MODEL.ADAPTATION: cocoopbatclip` | `model.py` (wrap with ClipTestTimePromptTuning, use_cocoop=True); `test_time.py` (registry lookup). |
| `MODEL.ARCH`, `WEIGHTS`, `USE_CLIP`, `UNIMODAL_IMAGE_ONLY` | `get_model`; `cocoop_batclip` (UNIMODAL_IMAGE_ONLY toggles I2T/InterMean). |
| `CLIP.PRECISION`, `FREEZE_TEXT_ENCODER`, `PROMPT_MODE`, `PROMPT_TEMPLATE` | CLIP/preprocess and ZeroShot path; CoCoOp path uses prompt_learner. |
| `TPT.SELECTION_P`, `N_CTX`, `CTX_INIT`, `CLASS_TOKEN_POS`, `LAMBDA_ENT` | `ClipTestTimePromptTuning` / CoCoOpPromptLearner; `cocoop_batclip` (selection_p, lambda_ent). |
| `OPTIM.METHOD`, `LR`, `WD`, `STEPS` | `TTAMethod.setup_optimizer` and steps in `forward`. |
| `CORRUPTION.*`, `TEST.*` | Data loader and evaluation in `test_time.py`. |

---

## 11. Summary

- **Entry:** `test_time.py` + `cfgs/cifar10_c/cocoop_batclip.yaml` and `conf.py`.
- **Method:** `methods/cocoop_batclip.py` → `CoCoOpBATCLIP` (subclass of `methods/base.TTAMethod`).
- **Model:** `models/model.get_model` → `models/custom_clip.ClipTestTimePromptTuning(use_cocoop=True)` → `CoCoOpPromptLearner` + `TextEncoder` + CLIP image encoder.
- **Losses:** `utils/losses.Entropy`, `I2TLoss`, `InterMeanLoss`; `methods/tpt.select_confident_samples`, `avg_entropy`.
- **Data:** `datasets/cls_names.get_class_names`; `datasets/data_loading.get_test_loader`.
- **Registry:** `utils/registry.ADAPTATION_REGISTRY`; registration in `methods/__init__.py`.
- **Not used for this path:** `models/cocoop.py` (standalone CoCoOp wrapper).

This is the full tree and flow for **cocoop_batclip** in this codebase.

