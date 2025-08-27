import os
import pathlib

# Set dataset path
dataset_path = r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data"
os.environ["RPPG_DATA_ROOT"] = dataset_path

# Test verification script
print("Testing dataset access...")
print(f"Dataset path: {dataset_path}")

pure_path = pathlib.Path(dataset_path) / "PURE"
print(f"PURE exists: {pure_path.exists()}")

if pure_path.exists():
    samples = list(pure_path.glob("*/*.json"))
    print(f"Found {len(samples)} JSON files")
    if samples:
        print(f"Sample file: {samples[0]}")

# Test package imports
print("\nTesting package imports...")
try:
    import torch
    print(f"OK PyTorch {torch.__version__}")
except ImportError as e:
    print(f"FAIL PyTorch: {e}")

try:
    import cv2
    print(f"OK OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"FAIL OpenCV: {e}")

try:
    import numpy as np
    print(f"OK NumPy {np.__version__}")
except ImportError as e:
    print(f"FAIL NumPy: {e}")

try:
    import mediapipe
    print(f"OK MediaPipe {mediapipe.__version__}")
except ImportError as e:
    print(f"FAIL MediaPipe: {e}")

print("\nSetup test completed!")