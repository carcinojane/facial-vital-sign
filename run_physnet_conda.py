"""
Run PhysNet using conda rppg environment
Workaround for OneDrive file access issues
"""
import subprocess
import sys

# Use conda run to execute in rppg environment
conda_exe = r"C:\Users\janej\anaconda3\Scripts\conda.exe"
toolbox_main = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\rppg_toolbox\main.py"
config_path = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\physnet_config_local.yaml"

cmd = [
    conda_exe, "run", "-n", "rppg",
    "python", toolbox_main,
    "--config_file", config_path
]

print(" ".join(cmd))
sys.exit(subprocess.call(cmd))
