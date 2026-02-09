"""
Train CoOp (context-only prompt tuning) on CIFAR-10.
Only prompt_learner.ctx is trained; no meta_net (unlike CoCoOp).
Checkpoint compatible with TPT/CoOp loading in get_model() (state_dict['ctx']).
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
    p = argparse.ArgumentParser(description="Train CoOp on CIFAR-10")
    p.add_argument("--arch", type=str, default="ViT-B-16")
    p.add_argument("--weights", type=str, default="openai")
    p.add_argument("--n_ctx", type=int, default=4, help="Number of context tokens")
    p.add_argument("--ctx_init", type=str, default="a photo of a", help="Context init (use _ for spaces)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"])
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
            logger.warning("Epoch %s: non-finite loss, skipping batch", epoch)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
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
    logger.info("Loading CLIP: %s, weights: %s", arch_for_clip, args.weights)
    clip_model, _, preprocess = create_model_and_transforms(
        arch_for_clip, pretrained=args.weights, device=device, precision=args.precision
    )
    normalization = preprocess.transforms[-1]
    preprocess.transforms = preprocess.transforms[:-1]

    # CoOp: use_cocoop=False -> PromptLearner only (ctx, no meta_net)
    logger.info("Creating CoOp model...")
    model = ClipTestTimePromptTuning(
        clip_model,
        normalization,
        arch_for_clip,
        "cifar10",
        n_ctx=args.n_ctx,
        ctx_init=args.ctx_init.replace(" ", "_"),
        class_token_pos="end",
        use_cocoop=False,  # CoOp: static context only
    )
    model = model.to(device)

    # Train only ctx (exclude token_embedding, which is registered as submodule of prompt_learner)
    for p in model.parameters():
        p.requires_grad = False
    trainable_params = [p for name, p in model.prompt_learner.named_parameters() if "token_embedding" not in name]
    for p in trainable_params:
        p.requires_grad = True
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info("Trainable parameters: %s", n_trainable)

    train_loader, test_loader = get_cifar10_loaders(args.data_dir, args.batch_size)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "coop_cifar10_best.pth")
    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, trainable_params)
        logger.info("Epoch %s: Train Loss: %.4f, Train Acc: %.2f%%", epoch, train_loss, train_acc)
        test_acc = evaluate(model, test_loader, device)
        logger.info("Test Acc: %.2f%%", test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "state_dict": model.prompt_learner.state_dict(),
                "epoch": epoch,
                "n_ctx": args.n_ctx,
                "arch": args.arch,
            }, save_path)
            logger.info("Saved best to %s", save_path)

    logger.info("Done. Best test acc: %.2f%%", best_acc)


if __name__ == "__main__":
    main()
