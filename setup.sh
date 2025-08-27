#!/usr/bin/env bash
set -e

# Create & activate env
if ! command -v conda &>/dev/null; then
  echo "Please install Miniconda/Anaconda first: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

conda env create -f env.yml || conda env update -f env.yml
eval "$(conda shell.bash hook)"
conda activate rppg

# Clone rPPG-Toolbox if missing
if [ ! -d "rppg_toolbox/.git" ]; then
  git clone https://github.com/ubicomplab/rPPG-Toolbox.git rppg_toolbox
fi

# Editable install (optional; so you can `import rppg_toolbox` locally)
cd rppg_toolbox
pip install -e . || true
cd ..

echo "✅ Setup complete. Next:"
echo "  1) Put your datasets in Google Drive (see README)"
echo "  2) python scripts/make_symlinks.py"
echo "  3) python scripts/verify_layout.py"
