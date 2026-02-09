import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
from tta_losses import Entropy, I2TLoss, InterMeanLoss

# Import setup_cfg from your existing train.py
from train import setup_cfg
from dassl.engine import build_trainer


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # shape [batch_size, num_classes]
    # Compute log probabilities
    log_probs = torch.log(probs + 1e-12)  # add epsilon to avoid log(0)
    # Compute entropy per sample
    entropy = -torch.sum(probs * log_probs, dim=-1)  # shape [batch_size]
    
    return entropy


def get_entropy_loss(outputs):
    """
    Computes Entropy Loss: H(x) = -sum(p(x) * log(p(x)))
    Minimizing this makes the model more confident in its predictions.
    """
    p = F.softmax(outputs, dim=1)
    return -(p * torch.log(p + 1e-6)).sum(dim=1).mean()

def collect_params(model):
    """
    Collects LayerNorm and Prompt Context parameters for optimization.
    """
    params = []
    names = []
    
    for name, param in model.named_parameters():
        # 1. Prompt Context Vectors (MaPLe specific)
        if "ctx"  or "compound_prompts_text" in name: 
            param.requires_grad = True
            params.append(param)
            names.append(name)
            
        # 2. LayerNorm Parameters
        elif "ln" in name or "LayerNorm" in name or "norm" in name:
            param.requires_grad = True
            params.append(param)
            names.append(name)
            
        else:
            # Freeze everything else
            param.requires_grad = False

    return params, names

def run_test_time_adaptation(trainer, args):
    """
    Main loop for Test-Time Adaptation
    """
    # 1. Setup Model and Data
    model = trainer.model
    # Ensure model is in eval mode (handles Dropout/BatchNorm correctly)
    model.eval() 
    
    # Get the test loader from the trainer
    data_loader = trainer.test_loader
    
    # Check what we are updating
    params, names = collect_params(model)
    print(f"✅ TTA Configured. Updating {len(names)} parameters (Prompts + LayerNorms).")
    
    # 2. Save the initial state (Anchor)
    # We use this to reset the model after every batch (Episodic TTA)
    anchor_state = copy.deepcopy(model.state_dict())

    # Metrics
    correct = 0
    total = 0

    print("🚀 Starting TTA Inference...")
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        input, label = trainer.parse_batch_test(batch)
        
        # --- RESET MODEL TO ANCHOR STATE ---
        # This ensures we adapt to the *current* image specifically
        # without carrying over noise from previous images.
        model.load_state_dict(anchor_state)
        
        # Define Optimizer for this specific episode
        # LR usually needs to be small for TTA (e.g., 1e-4 or 1e-5)
        optimizer = optim.SGD(params, lr=args.tta_lr, momentum=0.9)

        # --- ADAPTATION LOOP ---
        for step in range(args.tta_steps):
            optimizer.zero_grad()
            outputs = model(input)
            logits,image_features, text_features = outputs

            # loss = get_entropy_loss(outputs)
            loss = softmax_entropy(logits).mean(0)
            i2t_loss = i2t_loss(logits, image_features, text_features)
            inter_mean_loss = inter_mean_loss(logits, image_features)
            loss -= i2t_loss
            loss -= inter_mean_loss

            loss.backward()
            optimizer.step()
        
        # --- FINAL PREDICTION ---
        with torch.no_grad():
            outputs = model(input)
            _, pred = outputs.max(1)
            correct += pred.eq(label).sum().item()
            total += input.size(0)

    accuracy = 100 * correct / total
    print(f"🎉 TTA Finished. Final Accuracy: {accuracy:.2f}%")

# if __name__ == "__main__":
parser = argparse.ArgumentParser()
print("hello")
# --- Arguments required by train.py setup_cfg ---
parser.add_argument("--root", type=str, default="", help="path to dataset")
parser.add_argument("--output-dir", type=str, default="", help="output directory")
parser.add_argument("--resume", type=str, default="", help="path to model.pth.tar")
parser.add_argument("--seed", type=int, default=-1, help="random seed")
parser.add_argument("--source-domains", type=str, nargs="+", help="source domains")
parser.add_argument("--target-domains", type=str, nargs="+", help="target domains")
parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation")
parser.add_argument("--config-file", type=str, default="", help="path to config file")
parser.add_argument("--dataset-config-file", type=str, default="", help="dataset config")
parser.add_argument("--trainer", type=str, default="MaPLe", help="name of trainer")
parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
parser.add_argument("--head", type=str, default="", help="name of head")
parser.add_argument("--eval-only", action="store_true", help="evaluation only")
parser.add_argument("--model-dir", type=str, default="", help="load model from this directory")
parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch")
parser.add_argument("--no-train", action="store_true", help="do not train")
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options")

# --- TTA Specific Arguments ---
parser.add_argument("--tta-steps", type=int, default=1, help="Number of optimization steps per batch")
parser.add_argument("--tta-lr", type=float, default=1e-5, help="Learning rate for TTA")

args = parser.parse_args()

# 1. Setup Configuration using your existing function
cfg = setup_cfg(args)

# 2. Build Trainer (This creates the model and data loaders)
trainer = build_trainer(cfg)

# 3. Load the weights explicitly from the argument
if args.model_dir:
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
elif args.resume:
    print(f"Loading weights from: {args.resume}")
    trainer.load_model(args.resume)

# 4. Run TTA
run_test_time_adaptation(trainer, args)