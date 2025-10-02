# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-10-02

### Added

#### Documentation
- **DATASETS.md** - Comprehensive dataset download and setup guide
  - Direct links to PURE dataset on Kaggle (recommended)
  - UBFC-PHYS official page link
  - Step-by-step setup instructions
  - Expected directory structure
  - Citation requirements
  - Troubleshooting guide

- **GITHUB_PREP.md** - Complete GitHub preparation checklist
  - Pre-push verification steps
  - Safety checks for large files
  - Repository setup instructions
  - Common issues and solutions

- **data/README.md** - Placeholder documentation in data directory
  - Quick setup reference
  - Prevents empty directory in Git

- **Quick Links section in README.md**
  - Easy navigation to key documentation files

#### Evaluation Framework
- **scripts/rppg_processor_enhanced.py** - Enhanced rPPG processor
  - 7+ configurable improvement methods
  - Motion filtering
  - Multi-ROI (forehead + cheeks)
  - Signal detrending
  - Adaptive bandpass filtering
  - Temporal smoothing
  - Outlier rejection
  - Quality assessment (SNR)

- **run_incremental_evaluation.py** - Systematic improvement evaluation
  - Phase 1: Individual method testing
  - Phase 2: Cumulative combinations
  - Automated best-method selection
  - Comprehensive results.txt generation
  - Improvement delta calculations

#### Evaluation Scripts
- **run_pure_evaluation_optimized.py** - Optimized PURE evaluation
  - Frame skipping (2x speedup)
  - Incremental result saves
  - Progress tracking
  - No timeout issues

- **run_combined_evaluation.py** - Combined PURE + UBFC evaluation
  - Unified results across both datasets
  - Cross-dataset performance comparison

- **scripts/evaluate_pure.py** - PURE dataset evaluation module
  - Image sequence processing
  - JSON ground truth parsing
  - Comprehensive metrics

#### Dataset Management
- **quick_pure_test.py** - Fast 3-subject PURE test
- **medium_test.py** - Medium-scale testing
- **full_available_test.py** - Complete evaluation suite

### Updated

#### .gitignore
- Added comprehensive dataset exclusion patterns
  - `data/PURE/**`
  - `data/UBFC/**`
  - Video files (`*.avi`, `*.mp4`, `*.mov`, `*.zip`)
  - Result files (`*.csv`, `results.txt`, `*_evaluation_report*.txt`)
  - Plot images (`*.png`, `*.jpg`, `*.jpeg`)
  - Model checkpoints (`*.pt`, `*.pth`, `*.h5`)
- Preserves directory structure with `!data/README.md`

#### README.md
- Added dataset setup section with official download links
- Updated PURE dataset link (Kaggle + official TU Ilmenau)
- Updated UBFC dataset link (UBFC-PHYS main repository)
- Added clear warning about dataset size
- Added expected directory structure
- Added evaluation script documentation

#### CLAUDE.md
- Updated dataset configuration section
- Added instructions for symlink creation
- Documented evaluation workflow

### Changed
- Dataset links updated to most accessible sources
  - PURE: Kaggle (direct download) as primary
  - UBFC: UBFC-PHYS as primary source
- Evaluation methodology optimized for speed
  - Frame skipping for PURE (2x faster)
  - Efficient video processing for UBFC
  - Incremental saves to prevent data loss

### Fixed
- Timeout issues with long evaluations (background execution)
- Large file Git issues (comprehensive .gitignore)
- Dataset accessibility (Kaggle direct download)
- Documentation clarity (dedicated DATASETS.md)

### Security
- Added .gitignore patterns to prevent accidental commits of:
  - Large datasets (>20 GB)
  - Generated result files
  - Temporary outputs
  - Video files

## Repository Structure

```
rppg-vscode-starter/
├── scripts/
│   ├── evaluate_pure.py              # PURE dataset evaluation
│   ├── evaluate_ubfc.py              # UBFC dataset evaluation
│   ├── rppg_processor_enhanced.py    # Enhanced processor with improvements
│   └── ... (other scripts)
├── data/
│   ├── README.md                     # Dataset setup guide
│   ├── PURE/                         # [NOT INCLUDED] Download separately
│   └── UBFC/                         # [NOT INCLUDED] Download separately
├── run_incremental_evaluation.py    # Incremental improvement testing
├── run_pure_evaluation_optimized.py # Optimized PURE evaluation
├── run_combined_evaluation.py       # Combined dataset evaluation
├── DATASETS.md                       # Comprehensive dataset guide
├── GITHUB_PREP.md                    # GitHub preparation checklist
├── README.md                         # Main documentation
├── CLAUDE.md                         # AI development guidelines
├── .gitignore                        # Comprehensive exclusion patterns
└── CHANGELOG.md                      # This file
```

## Dataset Information

### PURE Dataset
- **Source**: [Kaggle](https://www.kaggle.com/datasets/computerscience3/public-requirementspure-dataset)
- **Size**: ~15 GB
- **Subjects**: 24 (10 subjects × 6 tasks, but 24 available in Kaggle version)

### UBFC-rPPG Dataset
- **Source**: [UBFC-PHYS](https://sites.google.com/view/ybenezeth/ubfc-phys)
- **Size**: ~8 GB
- **Subjects**: 42

## Evaluation Results

### Current Performance (Baseline POS Algorithm)
- **PURE Dataset** (24 subjects):
  - MAE: 4.569 ± 4.442 BPM
  - RMSE: 7.986 ± 7.090 BPM
  - Correlation: 0.323 ± 0.323
  - Within 10 BPM: 89.3% ± 16.0%

### Incremental Improvements
See `results.txt` for comprehensive incremental evaluation report showing:
- Individual method performance
- Best single method identification
- Cumulative combination results
- Overall improvement metrics

## Notes for Contributors

1. **Never commit datasets** - They are excluded via .gitignore
2. **Use incremental evaluation** - Test improvements systematically
3. **Document changes** - Update this changelog
4. **Cite datasets** - Required for publications (see DATASETS.md)
5. **Follow GitHub prep** - Use GITHUB_PREP.md before pushing

## Links

- [Dataset Setup Guide](DATASETS.md)
- [GitHub Preparation](GITHUB_PREP.md)
- [Main README](README.md)
- [Repository Description](REPOSITORY_DESCRIPTION.md)

---

**Version**: 1.0.0
**Date**: 2025-10-02
**Status**: Active Development
