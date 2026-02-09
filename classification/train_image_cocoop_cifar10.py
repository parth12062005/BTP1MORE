"""
Train text CoCoOp + image CoCoOp simultaneously on CIFAR-10.
- Text CoCoOp: image conditions text (image -> meta_net -> delta -> add to context -> image-conditioned prompts).
- Image CoCoOp: text conditions image (text_context -> reverse_meta_net -> delta -> add to image embedding).
- Both prompt_learner (ctx + meta_net) and reverse_meta_net are trained in the same forward/backward.
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
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from open_clip import create_model_and_transforms
from models.custom_clip import ClipTestTimePromptTuning

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train text CoCoOp + image CoCoOp simultaneously on CIFAR-10")
    p.add_argument("--arch", type=str, default="ViT-B-16", help="CLIP arch (open_clip: ViT-B-16)")
    p.add_argument("--weights", type=str, default="openai", help="CLIP pretrained weights")
    p.add_argument("--n_ctx", type=int, default=4, help="Number of context tokens")
    p.add_argument("--ctx_init", type=str, default="a photo of a", help="Context init phrase (use _ for spaces)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save_dir", type=str, default="./checkpoints", help="Dir to save best checkpoint")
    p.add_argument("--text_ckpt", type=str, default="text_cocoop_cifar10_best.pth", help="Checkpoint filename for text CoCoOp (prompt_learner)")
    p.add_argument("--image_ckpt", type=str, default="image_cocoop_cifar10_best.pth", help="Checkpoint filename for image CoCoOp (reverse_meta_net)")
    p.add_argument("--data_dir", type=str, default="./data", help="CIFAR-10 root")
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"], help="Training precision")
    return p.parse_args()


def get_cifar10_loaders(data_dir, batch_size):
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

    arch_for_clip = args.arch if "-" in args.arch else args.arch.replace("/", "-")
    logger.info("Loading CLIP model: %s with weights: %s", arch_for_clip, args.weights)
    clip_model, _, preprocess = create_model_and_transforms(
        arch_for_clip, pretrained=args.weights, device=device, precision=args.precision
    )
    normalization = preprocess.transforms[-1]
    preprocess.transforms = preprocess.transforms[:-1]

    logger.info("Creating model: text CoCoOp + image CoCoOp (both train simultaneously)...")
    model = ClipTestTimePromptTuning(
        clip_model,
        normalization,
        arch_for_clip,
        "cifar10",
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init.replace(" ", "_"),
        class_token_pos="end",
        use_cocoop=True,        # text CoCoOp: image conditions text (meta_net)
        use_reverse_cocoop=True,  # image CoCoOp: text conditions image (reverse_meta_net)
    )
    model = model.to(device)

    # Train both: prompt_learner (ctx + meta_net) and reverse_meta_net
    for p in model.parameters():
        p.requires_grad = False
    trainable_params = []
    for name, p in model.prompt_learner.named_parameters():
        if "token_embedding" not in name:
            p.requires_grad = True
            trainable_params.append(p)
    for p in model.reverse_meta_net.parameters():
        p.requires_grad = True
        trainable_params.append(p)
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters (prompt_learner + reverse_meta_net): %s / %s (%.3f%%)", n_trainable, n_total, 100.0 * n_trainable / n_total)

    train_loader, test_loader = get_cifar10_loaders(args.data_dir, args.batch_size)
    logger.info("Loading CIFAR-10 dataset...")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup + ReduceLROnPlateau
    warmup_epochs = 5
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path_text = os.path.join(args.save_dir, args.text_ckpt)
    save_path_image = os.path.join(args.save_dir, args.image_ckpt)
    start_epoch = 0
    best_acc = -1.0

    # Load from separate checkpoints if available (text CoCoOp and image CoCoOp)
    if os.path.isfile(save_path_text):
        logger.info("Loading text CoCoOp from %s", save_path_text)
        ckpt = torch.load(save_path_text, map_location=device)
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            pl_sd = model.prompt_learner.state_dict()
            pl_load = {k: v for k, v in sd.items() if k in pl_sd and k not in ("token_prefix", "token_suffix") and pl_sd[k].shape == v.shape}
            if pl_load:
                model.prompt_learner.load_state_dict(pl_load, strict=False)
                logger.info("Loaded %s keys into text CoCoOp (prompt_learner).", len(pl_load))
            start_epoch = ckpt.get("epoch", start_epoch)
            best_acc = ckpt.get("best_acc", best_acc)
        else:
            logger.warning("Text checkpoint has no 'state_dict'.")
    else:
        logger.info("No text CoCoOp checkpoint at %s; starting text CoCoOp from scratch.", save_path_text)

    if os.path.isfile(save_path_image):
        logger.info("Loading image CoCoOp from %s", save_path_image)
        ckpt = torch.load(save_path_image, map_location=device)
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            rmn_sd = model.reverse_meta_net.state_dict()
            rmn_load = {k: v for k, v in sd.items() if k in rmn_sd and rmn_sd[k].shape == v.shape}
            if rmn_load:
                model.reverse_meta_net.load_state_dict(rmn_load, strict=False)
                logger.info("Loaded %s keys into image CoCoOp (reverse_meta_net).", len(rmn_load))
            start_epoch = max(start_epoch, ckpt.get("epoch", 0))
            best_acc = max(best_acc, ckpt.get("best_acc", -1.0))
        else:
            logger.warning("Image checkpoint has no 'state_dict'.")
    else:
        logger.info("No image CoCoOp checkpoint at %s; starting image CoCoOp from scratch.", save_path_image)

    if start_epoch > 0 or best_acc > -1.0:
        logger.info("Resuming from epoch %s, best_acc so far: %.2f%%", start_epoch, best_acc)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Warmup scheduler
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor
            logger.info("Warmup epoch %s/%s, LR: %.6f", epoch, warmup_epochs, optimizer.param_groups[0]['lr'])
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, trainable_params)
        logger.info("Epoch %s: Train Loss: %.4f, Train Acc: %.2f%%", epoch, train_loss, train_acc)
        test_acc = evaluate(model, test_loader, device)
        logger.info("Test Acc: %.2f%%", test_acc)
        
        # ReduceLROnPlateau step (after warmup)
        if epoch > warmup_epochs:
            scheduler.step(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save text CoCoOp and image CoCoOp in separate checkpoints
            meta = {"epoch": epoch, "best_acc": best_acc, "arch": args.arch}
            torch.save({"state_dict": model.prompt_learner.state_dict(), **meta}, save_path_text)
            torch.save({"state_dict": model.reverse_meta_net.state_dict(), **meta}, save_path_image)
            logger.info("Saved text CoCoOp to %s and image CoCoOp to %s (acc: %.2f%%)", save_path_text, save_path_image, best_acc)

    logger.info("Done. Best test acc: %.2f%%", best_acc)


if __name__ == "__main__":
    main()