"""
Run PhysNet fine-tuning on mobile phone video dataset using conda environment.

This script fine-tunes a pre-trained PhysNet model on mobile phone videos.
"""
import subprocess
import os
import sys

# Paths
conda_exe = r"C:\Users\janej\anaconda3\Scripts\conda.exe"
toolbox_main = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\rppg_toolbox\main.py"
config_path = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\physnet_mobile_finetune.yaml"

# Check if config file exists
if not os.path.exists(config_path):
    print(f"Error: Configuration file not found: {config_path}")
    sys.exit(1)

# Change to the toolbox directory
toolbox_dir = os.path.dirname(toolbox_main)

# Build conda run command
cmd = [
    conda_exe, "run",
    "-n", "rppg",  # conda environment name
    "python", toolbox_main,
    "--config_file", config_path
]

print("="*70)
print("PhysNet Mobile Fine-tuning")
print("="*70)
print(f"Config: {config_path}")
print(f"Working directory: {toolbox_dir}")
print(f"\nCommand: {' '.join(cmd)}")
print("="*70)
print("\nStarting fine-tuning...")
print("This may take several hours depending on dataset size and hardware.")
print("\nMonitor:")
print("  - Training loss should decrease over epochs")
print("  - Validation loss should decrease and plateau")
print("  - Best model will be saved based on lowest validation loss")
print("="*70)
print()

# Run the command
try:
    result = subprocess.run(
        cmd,
        cwd=toolbox_dir,
        capture_output=False,  # Show output in real-time
        text=True
    )

    print(f"\n{'='*70}")
    if result.returncode == 0:
        print("Fine-tuning completed successfully!")
        print("\nNext steps:")
        print("  1. Check runs/mobile_finetune/ for trained models")
        print("  2. Review training logs and loss curves")
        print("  3. Test on mobile videos using: python scripts/evaluate_mobile_model.py")
    else:
        print(f"Fine-tuning failed with exit code: {result.returncode}")
        print("Check error messages above for details")
    print("="*70)

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    print("Partial models may be saved in runs/mobile_finetune/")
    sys.exit(1)
except Exception as e:
    print(f"\nError running fine-tuning: {e}")
    sys.exit(1)
