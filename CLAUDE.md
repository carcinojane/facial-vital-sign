# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an rPPG (remote photoplethysmography) research project for heart rate estimation from face videos. It's designed as a VS Code starter template that integrates with Google Drive for dataset storage and uses the rPPG-Toolbox for advanced training/evaluation.

## Environment Setup

The project uses Conda for environment management:

```bash
# Initial setup
bash setup.sh
# OR on Windows PowerShell:
conda env create -f env.yml
conda activate rppg
git clone https://github.com/ubicomplab/rPPG-Toolbox.git rppg_toolbox
```

**Important**: The `rppg` conda environment must be activated before running any scripts. The environment includes PyTorch, OpenCV, MediaPipe, and CUDA toolkit (remove `cudatoolkit=11.8` from env.yml for CPU-only).

## Dataset Management

The project expects datasets in Google Drive at path `G:\My Drive\rppg_datasets` (Windows) or set `GDRIVE_RPPG` environment variable to customize:

```bash
# Link Google Drive datasets to local project
python scripts/make_symlinks.py
python scripts/verify_layout.py
```

Expected structure:
- `data/PURE/` - PURE dataset (symlinked from Google Drive)
- `data/UBFC-rPPG/` - UBFC-rPPG dataset (symlinked from Google Drive)
- `rppg_toolbox/` - Cloned rPPG-Toolbox repository

## Core Scripts

### Quick Heart Rate Estimation
```bash
# Run unsupervised POS algorithm on any video
python scripts/run_unsupervised_pos.py --video ./sample_video.mp4 [--fps 30]
```

### rPPG-Toolbox Integration
```bash
# Run official toolbox with specific config
python scripts/run_toolbox_main.py --config configs/eval_configs/<config_name.yaml>
```

## Architecture

**scripts/run_unsupervised_pos.py**: Implements POS (Plane-Orthogonal-to-Skin) algorithm for real-time heart rate estimation:
- Uses MediaPipe for face detection/landmarks
- Extracts RGB signals from forehead/cheek regions
- Applies bandpass filtering (0.7-3.0 Hz)
- Estimates HR via frequency domain analysis

**scripts/make_symlinks.py**: Creates symbolic links from Google Drive datasets to local `data/` directory. Handles Windows permission issues gracefully.

**scripts/run_toolbox_main.py**: Wrapper for rPPG-Toolbox main.py, allowing relative config paths and proper working directory setup.

**scripts/verify_layout.py**: Validates expected dataset structure exists and contains required files (JSON for PURE, ground_truth.txt for UBFC).

## Development Environment

- Python 3.10 with conda environment isolation
- PyTorch ecosystem for deep learning components
- OpenCV + MediaPipe for computer vision
- Designed for both CPU and GPU (CUDA 11.8) execution
- Google Drive integration via local mount/symlinks

## Platform Considerations

- **Windows**: May require Developer Mode or Administrator privileges for symlink creation
- **Cross-platform**: Uses pathlib for path handling, supports both Windows and Unix-style paths
- **Google Drive**: Expects Google Drive for Desktop or rclone mount for dataset access