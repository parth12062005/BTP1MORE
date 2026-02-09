"""
Standalone script to train CoCoOp on CIFAR-10 (supervised).
Trains only prompt learner (ctx + meta_net). Saves checkpoint compatible with
CoCoOp-BATCLIP TTA loading in get_model() (state_dict: ctx, meta_net with linear1, norm1, linear2, norm2, linear3).
"""
import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from open_clip import create_model_and_transforms
from models.custom_clip import ClipTestTimePromptTuning

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train CoCoOp on CIFAR-10")
    p.add_argument("--arch", type=str, default="ViT-B-16", help="CLIP arch (open_clip: ViT-B-16)")
    p.add_argument("--weights", type=str, default="openai", help="CLIP pretrained weights")
    p.add_argument("--n_ctx", type=int, default=4, help="Number of context tokens")
    p.add_argument("--ctx_init", type=str, default="a photo of a", help="Context init phrase (use _ for spaces)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save_dir", type=str, default="./checkpoints", help="Dir to save best checkpoint")
    p.add_argument("--ckpt_name", type=str, default="cocoop_cifar10_best.pth", help="Checkpoint filename to save/load")
    p.add_argument("--data_dir", type=str, default="./data", help="CIFAR-10 root")
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"], help="Training precision")
    return p.parse_args()


def get_cifar10_loaders(data_dir, batch_size):
    # CLIP ViT expects 224x224; CIFAR-10 is 32x32 -> resize. Model applies normalize in forward().
    train_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_ds = CIFAR10(root=data_dir, train=True, download=True, transform=train_t)
    test_ds = CIFAR10(root=data_dir, train=False, download=True, transform=test_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device, epoch, trainable_params):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        logits = logits.clamp(-50.0, 50.0)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss).all():
            logger.warning("Epoch %s: non-finite loss detected, skipping batch", epoch)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * total_correct / total:.2f}%")
    return total_loss / max(len(loader), 1), 100.0 * total_correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Arch for create_model: open_clip uses "ViT-B-16" (hyphen). Our custom_clip tokenizer expects same.
    arch_for_clip = args.arch if "-" in args.arch else args.arch.replace("/", "-")
    logger.info("Loading CLIP model: %s with weights: %s", arch_for_clip, args.weights)
    clip_model, _, preprocess = create_model_and_transforms(
        arch_for_clip, pretrained=args.weights, device=device, precision=args.precision
    )
    normalization = preprocess.transforms[-1]
    preprocess.transforms = preprocess.transforms[:-1]

    logger.info("Creating CoCoOp model...")
    model = ClipTestTimePromptTuning(
        clip_model,
        normalization,
        arch_for_clip,
        "cifar10",
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init.replace(" ", "_"),
        class_token_pos="end",
        use_cocoop=True,
    )
    model = model.to(device)
    # CoCoOp uses 3-layer meta_net + LayerNorm (defined in custom_clip.CoCoOpPromptLearner)

    # Train only prompt learner: ctx + meta_net (exclude token_embedding, which is registered as submodule)
    for p in model.parameters():
        p.requires_grad = False
    trainable_params = []
    for name, p in model.prompt_learner.named_parameters():
        if "token_embedding" not in name:
            p.requires_grad = True
            trainable_params.append(p)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %s / %s (%.3f%%)", n_trainable, n_total, 100.0 * n_trainable / n_total)

    train_loader, test_loader = get_cifar10_loaders(args.data_dir, args.batch_size)
    logger.info("Loading CIFAR-10 dataset...")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.ckpt_name)
    start_epoch = 0
    best_acc = -1.0

    # If existing CoCoOp checkpoint exists, load it and resume training (improve from that point)
    if os.path.isfile(save_path):
        logger.info("Found existing checkpoint %s; loading and resuming training.", save_path)
        ckpt = torch.load(save_path, map_location=device)
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            # Strip keys that may not match (token_prefix, token_suffix are class-name dependent)
            sd = {k: v for k, v in sd.items() if k not in ("token_prefix", "token_suffix")}
            pl_sd = model.prompt_learner.state_dict()
            to_load = {k: v for k, v in sd.items() if k in pl_sd and pl_sd[k].shape == v.shape}
            if to_load:
                model.prompt_learner.load_state_dict(to_load, strict=False)
                logger.info("Loaded %s keys from checkpoint into prompt_learner.", len(to_load))
            start_epoch = ckpt.get("epoch", 0)
            best_acc = ckpt.get("best_acc", -1.0)
            logger.info("Resuming from epoch %s, best_acc so far: %.2f%%", start_epoch, best_acc)
        else:
            logger.warning("Checkpoint has no 'state_dict'; starting from scratch.")
    else:
        logger.info("No existing checkpoint at %s; starting new training.", save_path)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, trainable_params)
        logger.info("Epoch %s: Train Loss: %.4f, Train Acc: %.2f%%", epoch, train_loss, train_acc)
        test_acc = evaluate(model, test_loader, device)
        logger.info("Test Acc: %.2f%%", test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                "state_dict": model.prompt_learner.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "n_ctx": args.n_ctx,
                "arch": args.arch,
            }
            torch.save(state, save_path)
            logger.info("Saved best checkpoint to %s (acc: %.2f%%)", save_path, best_acc)

    logger.info("Done. Best test acc: %.2f%%", best_acc)


if __name__ == "__main__":
    main()
