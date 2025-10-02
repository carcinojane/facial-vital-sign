# Dataset Directory

This directory is for storing research datasets.

## ⚠️ Important

**Datasets are NOT included in this repository.**

Please download datasets separately from their official sources. See `../DATASETS.md` for detailed instructions.

## Expected Structure

After downloading and extracting datasets, this directory should contain:

```
data/
├── PURE/
│   ├── 01-01/
│   ├── 01-02/
│   └── ... (24 subjects total)
│
├── UBFC/
│   ├── subject1/
│   ├── subject2/
│   └── ... (42 subjects total)
│
└── README.md (this file)
```

## Quick Setup

```bash
# 1. Download datasets (see DATASETS.md for links)
# 2. Extract to this directory
# 3. Verify structure:
python scripts/verify_layout.py
```

## Dataset Information

- **PURE Dataset**: ~15 GB, 10 subjects, image sequences
- **UBFC-rPPG Dataset**: ~8 GB, 42 subjects, video files

See `../DATASETS.md` for complete download and setup instructions.
