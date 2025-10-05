"""
Run PhysNet evaluation on UBFC dataset using conda environment
"""
import subprocess
import os

# Paths
conda_exe = r"C:\Users\janej\anaconda3\Scripts\conda.exe"
toolbox_main = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\rppg_toolbox\main.py"
config_path = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\physnet_ubfc_config.yaml"

# Change to the toolbox directory
toolbox_dir = os.path.dirname(toolbox_main)

# Build conda run command
cmd = [
    conda_exe, "run",
    "-n", "rppg",  # conda environment name
    "python", toolbox_main,
    "--config_file", config_path
]

print("Running PhysNet evaluation on UBFC dataset...")
print(f"Command: {' '.join(cmd)}")
print(f"Working directory: {toolbox_dir}")
print()

# Run the command
result = subprocess.run(
    cmd,
    cwd=toolbox_dir,
    capture_output=False,  # Show output in real-time
    text=True
)

print(f"\nProcess exited with code: {result.returncode}")
