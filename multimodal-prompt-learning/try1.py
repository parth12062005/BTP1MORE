import torch

# Path to your model file
model_path = "model.pth.tar-2"

# Load checkpoint
checkpoint = torch.load(model_path, map_location="cpu")
print(checkpoint.keys())


# CASE 2: If the file is a checkpoint dictionary
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Print parameter names and dimensions
print("Parameter Name -> Shape")
print("-" * 40)

for name, param in state_dict.items():
    print(f"{name:40s} {tuple(param.shape)}")
