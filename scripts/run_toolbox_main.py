import argparse, subprocess, sys, pathlib, os

ROOT = pathlib.Path(__file__).resolve().parents[1]
TBX = ROOT / "rppg_toolbox"

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="Relative path under rPPG-Toolbox/configs/")
args = ap.parse_args()

cfg_path = TBX / args.config
if not cfg_path.exists():
    raise SystemExit(f"Config not found: {cfg_path}\nList configs under: {TBX/'configs'}")
cmd = [sys.executable, str(TBX / "main.py"), "--config", str(cfg_path)]
print(">>", " ".join(cmd))
sys.exit(subprocess.call(cmd, cwd=str(TBX)))
