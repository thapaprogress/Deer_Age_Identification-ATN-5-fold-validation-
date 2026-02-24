import torch
import torchvision

print("="*60)
print("PYTORCH VERIFICATION")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
else:
    print("CUDA version: N/A (CPU only)")

print(f"Device to use: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("="*60)
