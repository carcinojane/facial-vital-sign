# 📊 Dataset Setup Guide

This document provides comprehensive instructions for obtaining and setting up the datasets required for this rPPG research project.

## ⚠️ Important Notice

**Datasets are NOT included in this repository due to their large size (>20 GB combined).**

All datasets used in this project are publicly available for research purposes. You must download them separately from their official sources.

---

## 📦 Required Datasets

### 1. PURE Dataset

**Remote Photoplethysmography Dataset for Heart Rate Estimation**

- **Kaggle (Direct Download)**: [PURE Dataset on Kaggle](https://www.kaggle.com/datasets/computerscience3/public-requirementspure-dataset)
- **Official Page**: [PURE Dataset - TU Ilmenau](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
- **Size**: ~15 GB
- **Subjects**: 10 individuals
- **Format**: Image sequences (.png) + JSON ground truth
- **Characteristics**:
  - 640×480 resolution
  - ~30 FPS
  - 6 different tasks per subject
  - Ground truth from pulse oximeter

**Download Instructions (Kaggle - Recommended):**
1. Visit [https://www.kaggle.com/datasets/computerscience3/public-requirementspure-dataset](https://www.kaggle.com/datasets/computerscience3/public-requirementspure-dataset)
2. Click "Download" button (requires Kaggle account - free)
3. Extract the downloaded zip file
4. Move/copy the extracted folders to `data/PURE/`

**Alternative (Official Source):**
1. Visit the TU Ilmenau official page
2. Read and accept the research license
3. Fill out the download request form
4. Download the dataset zip file
5. Extract to `data/PURE/`

**Expected Structure:**
```
data/PURE/
├── 01-01/
│   ├── 01-01.json              # Ground truth HR data
│   └── 01-01/                  # Image sequence folder
│       ├── Image1392643993642815000.png
│       ├── Image1392643993676160000.png
│       └── ...
├── 01-02/
├── 01-03/
...
├── 10-05/
└── 10-06/
```

### 2. UBFC-rPPG Dataset

**University of Burgundy Franche-Comté Remote Photoplethysmography Dataset**

- **Official Page (UBFC-PHYS)**: [https://sites.google.com/view/ybenezeth/ubfc-phys](https://sites.google.com/view/ybenezeth/ubfc-phys)
- **Alternative - UBFC-rPPG**: [https://sites.google.com/view/ybenezeth/ubfcrppg](https://sites.google.com/view/ybenezeth/ubfcrppg)
- **Size**: ~8 GB
- **Subjects**: 42 individuals (originally 49, some excluded)
- **Format**: Video files (.avi) + TXT ground truth
- **Characteristics**:
  - Uncompressed 8-bit RGB format
  - Various lighting conditions
  - Continuous HR ground truth from CMS50E pulse oximeter

**Download Instructions:**
1. Visit [https://sites.google.com/view/ybenezeth/ubfc-phys](https://sites.google.com/view/ybenezeth/ubfc-phys)
2. Look for the "Dataset" or "Download" section
3. Follow the download instructions provided on the page
4. Download the dataset archive
5. Extract to `data/UBFC/`

**Note**: UBFC-PHYS is the main repository. If you need the original UBFC-rPPG dataset specifically, use the alternative link above.

**Expected Structure:**
```
data/UBFC/
├── subject1/
│   ├── vid.avi                 # Video recording
│   └── ground_truth.txt        # HR values (two lines: PPG signal, HR)
├── subject2/
├── subject3/
...
└── subject49/
```

---

## 🛠️ Setup Instructions

### Step 1: Download Datasets

Download both datasets from their official sources as described above.

### Step 2: Create Directory Structure

```bash
cd rppg-vscode-starter

# Create data directories
mkdir -p data/PURE
mkdir -p data/UBFC
```

### Step 3: Extract Datasets

```bash
# Extract PURE dataset
unzip PURE.zip -d data/PURE/

# Extract UBFC dataset
unzip UBFC-rPPG.zip -d data/UBFC/
```

### Step 4: Verify Installation

Run the verification script to ensure datasets are properly structured:

```bash
python scripts/verify_layout.py
```

Expected output:
```
✓ PURE dataset found: 24 subjects
✓ UBFC dataset found: 42 subjects
✓ All ground truth files present
✓ Dataset structure verified
```

---

## 🔗 Alternative: Cloud Storage Setup

For collaborative work or if you're storing datasets on Google Drive:

### Option A: Google Drive (Recommended for Teams)

```bash
# 1. Install Google Drive for Desktop or rclone

# 2. Set environment variable
# Windows:
set GDRIVE_RPPG=G:\My Drive\rppg_datasets

# macOS/Linux:
export GDRIVE_RPPG=~/Google\ Drive/rppg_datasets

# 3. Create symbolic links
python scripts/make_symlinks.py
```

### Option B: External Drive

If datasets are on an external drive:

```bash
# Create symlinks to external drive
ln -s /Volumes/ExternalDrive/rppg_datasets/PURE data/PURE
ln -s /Volumes/ExternalDrive/rppg_datasets/UBFC data/UBFC
```

---

## 📈 Dataset Statistics

### PURE Dataset
| Metric | Value |
|--------|-------|
| Total Subjects | 10 |
| Total Tasks | 60 (6 per subject) |
| Resolution | 640×480 |
| Frame Rate | ~30 FPS |
| Duration/Task | ~1 minute |
| Total Frames | ~108,000 |
| Ground Truth | Pulse oximeter (Nonin 9560) |

### UBFC-rPPG Dataset
| Metric | Value |
|--------|-------|
| Total Subjects | 42 |
| Resolution | Varies (mostly 640×480) |
| Format | Uncompressed AVI |
| Duration/Subject | ~1-2 minutes |
| Ground Truth | CMS50E pulse oximeter |
| Special Features | Varied lighting conditions |

---

## 🔬 Using the Datasets

### Quick Evaluation

```bash
# Test on PURE dataset (first 3 subjects)
python quick_pure_test.py

# Test on UBFC dataset
python quick_test.py

# Full evaluation on both datasets
python run_combined_evaluation.py
```

### Incremental Improvement Testing

```bash
# Run systematic improvement evaluation
python run_incremental_evaluation.py
```

This will test different rPPG improvement methods and document their impact in `results.txt`.

---

## 📝 Citation Requirements

If you use these datasets in your research, please cite the original papers:

### PURE Dataset
```bibtex
@inproceedings{stricker2014non,
  title={Non-contact video-based pulse rate measurement on a mobile service robot},
  author={Stricker, Ronny and M{\"u}ller, Steffen and Gross, Horst-Michael},
  booktitle={The 23rd IEEE International Symposium on Robot and Human Interactive Communication},
  pages={1056--1062},
  year={2014},
  organization={IEEE}
}
```

### UBFC-rPPG Dataset
```bibtex
@inproceedings{bobbia2019unsupervised,
  title={Unsupervised skin tissue segmentation for remote photoplethysmography},
  author={Bobbia, S and Macwan, R and Benezeth, Y and Mansouri, A and Dubois, J},
  booktitle={Pattern Recognition and Image Analysis: 9th Iberian Conference},
  pages={303--308},
  year={2019}
}
```

---

## ❓ Troubleshooting

### Problem: "Dataset not found" error

**Solution:**
- Verify datasets are in `data/PURE/` and `data/UBFC/`
- Check directory names match exactly (case-sensitive)
- Run `python scripts/verify_layout.py`

### Problem: Large file size prevents Git operations

**Solution:**
- `.gitignore` is configured to exclude datasets
- Never use `git add .` - add specific files only
- Use `git status` to verify no large files are staged

### Problem: Out of disk space

**Solution:**
- PURE: ~15 GB free space required
- UBFC: ~8 GB free space required
- Total: Ensure at least 30 GB available (including processing space)

### Problem: Permission denied errors

**Solution:**
```bash
# On Unix/Linux/macOS:
chmod -R 755 data/

# On Windows:
# Right-click → Properties → Security → Edit permissions
```

---

## 📧 Support

For dataset-specific issues:
- **PURE**: Contact TU Ilmenau team via their official page
- **UBFC**: Contact University of Burgundy team

For code/implementation issues:
- Open an issue on this repository
- Include your Python version and error messages

---

## 🔒 License & Usage

Both datasets are provided for **research and educational purposes only**. Commercial use requires separate licensing from the dataset creators.

By downloading and using these datasets, you agree to:
- Use them only for research/educational purposes
- Cite the original papers in any publications
- Not redistribute the datasets without permission
- Respect privacy and ethical considerations

---

**Last Updated:** 2025-10-02

For the latest information, always check the official dataset pages.
