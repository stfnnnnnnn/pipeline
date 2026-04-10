import sys
import importlib

def check_module(module_name, display_name=None):
    name = display_name or module_name
    try:
        mod = importlib.import_module(module_name)
        # Try to get version, default to 'unknown' if not available
        version = getattr(mod, '__version__', 'unknown version')
        print(f"✅ {name} successfully imported (v{version})")
        return True
    except ImportError as e:
        print(f"❌ ERROR: Could not import {name}. ({e})")
        return False

print("=== MoodPlay Environment Sanity Check ===\n")

# 1. Check PyTorch & CUDA (The absolute most important part)
print("--- Checking PyTorch & GPU ---")
if check_module('torch', 'PyTorch'):
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"   GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   PyTorch CUDA Version Built With: {torch.version.cuda}")
    else:
        print("   ⚠️ WARNING: PyTorch cannot see your GPU. Processing will be extremely slow.")

# 2. Check Core Diffusion & Rendering Libraries
print("\n--- Checking Core Libraries ---")
check_module('diffusers')
check_module('transformers')
check_module('xformers')
check_module('ultralytics', 'YOLO (Ultralytics)')
check_module('cv2', 'OpenCV')

# 3. Check Custom Meta Repositories
print("\n--- Checking Meta Tracking Libraries ---")
check_module('sam2')
check_module('cotracker')

print("\n=========================================")
print("If all items have a ✅, your environment is ready to go!")