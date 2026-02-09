"""
Evaluate CLIP zero-shot accuracy on CIFAR-10.
Uses open_clip + same preprocessing as the rest of the codebase.
"""
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from open_clip import create_model_and_transforms, get_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="ViT-B-16")
    p.add_argument("--weights", type=str, default="openai")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"])
    return p.parse_args()


# CIFAR-10 class names (same as in datasets/cls_names.py)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and preprocess
    model, _, preprocess = create_model_and_transforms(
        args.arch, pretrained=args.weights, device=device, precision=args.precision
    )
    model.eval()
    tokenizer = get_tokenizer(args.arch)

    # Text prompts: "a photo of a {class}"
    prompts = [f"a photo of a {c}." for c in CIFAR10_CLASSES]
    text_tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Data: resize 32x32 -> 224x224 for ViT, then CLIP preprocess (normalize)
    transform = transforms.Compose([transforms.Resize((224, 224)), preprocess])
    test_ds = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).float()
            pred = logits.argmax(dim=1)
            correct += (pred == labels.to(device)).sum().item()
            total += labels.size(0)

    acc = 100.0 * correct / total
    print(f"CIFAR-10 zero-shot accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
