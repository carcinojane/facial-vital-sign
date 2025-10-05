# 🩺 Facial Vital Signs Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-Research-orange.svg)](LICENSE)

> **Transform any camera into a comprehensive vital signs monitoring system using facial video analysis**

## 📌 Quick Links

### 🎯 Essential Documentation
- **[Iteration History](ITERATIONS.md)** - Complete development log (Iterations 0-4)
- **[Evaluation Results](EVALUATION_RESULTS.md)** - Performance data and analysis
- **[Dataset Setup](DATASETS.md)** - Download & configure PURE/UBFC datasets
- **[Memory Optimization](MEMORY_OPTIMIZATION.md)** - Technical fixes and optimizations
- **[Changelog](CHANGELOG.md)** - Recent changes and updates

### 🔧 Technical Details
- **[Repository Documentation](REPOSITORY_DESCRIPTION.md)** - Full architecture details
- **[Claude.ai Instructions](CLAUDE.md)** - AI development guidelines

A state-of-the-art **remote photoplethysmography (rPPG)** system that extracts multiple physiological parameters from facial video streams without any contact or specialized hardware.

![Demo](https://via.placeholder.com/800x400?text=Facial+Vital+Signs+Demo)

## ✨ Features

### 🫀 Comprehensive Vital Signs
- **Heart Rate (HR)** - Real-time detection using POS algorithm
- **Respiratory Rate (RR)** - Breathing pattern analysis  
- **Blood Oxygen (SpO2)** - Oxygen saturation estimation
- **Heart Rate Variability (HRV)** - Cardiovascular health metrics
- **Blood Pressure (BP)** - Systolic/diastolic estimation
- **Stress Level** - Multi-parameter psychological assessment

### 🎨 Multiple Interfaces
- **Advanced UI** - Comprehensive monitoring with color-coded indicators
- **Dashboard** - Real-time graphs and data visualization
- **Basic UI** - Simple heart rate monitoring
- **Research Tools** - Dataset evaluation and performance analysis

### 🔬 Research Ready
- **UBFC-rPPG Dataset** support with automated evaluation
- **PURE Dataset** integration for benchmarking
- Statistical analysis with MAE, RMSE, MAPE metrics
- Export capabilities for further analysis

## 🚀 Quick Start

### 📋 Prerequisites
- Python 3.10+
- Miniconda/Anaconda
- Webcam or video files
- (Optional) CUDA-compatible GPU

### ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/carcinojane/facial-vital-sign.git
cd facial-vital-sign

# Create conda environment
conda env create -f env.yml
conda activate rppg

# Install dependencies
pip install -r requirements.txt
```

### 🎯 Basic Usage

```bash
# Launch advanced vital signs monitoring
python scripts/advanced_vitals_ui.py

# Open full dashboard with graphs
python scripts/vitals_dashboard.py

# Run quick functionality test
python test_enhanced_ui_simple.py
```

## 🎮 Demo Interfaces

### 1️⃣ Advanced Vital Signs UI
Real-time monitoring with comprehensive vital signs display:
```bash
python scripts/advanced_vitals_ui.py
```
**Features:**
- Multi-vital signs display (HR, RR, SpO2, HRV, BP, Stress)
- Color-coded health indicators
- Face detection with multiple ROI
- Signal quality assessment
- Screenshot and data saving

### 2️⃣ Full Dashboard
Complete monitoring dashboard with graphs:
```bash
python scripts/vitals_dashboard.py
```
**Features:**
- Real-time matplotlib visualizations
- Session statistics and trends
- Data export (CSV + summary)
- Pause/resume functionality
- Comprehensive vital signs history

### 3️⃣ Interactive Demo Menu
Comprehensive demo with multiple options:
```bash
python test_enhanced_ui.py
```
Choose from:
- Basic advanced UI demonstration
- Full dashboard with all features
- Component testing and validation
- Feature comparison matrix

## 🔧 Configuration

### Environment Setup
```bash
# CPU-only installation (remove CUDA)
conda env create -f env_cpu.yml

# Custom dataset path
export GDRIVE_RPPG="/path/to/your/datasets"

# GPU acceleration (default)
# Requires CUDA 11.8+ drivers
```

### Dataset Integration

**⚠️ IMPORTANT: Datasets are NOT included in this repository due to size constraints**

This project uses the following public datasets for evaluation:

#### 📦 Required Datasets

1. **PURE Dataset**
   - **Kaggle (Recommended)**: [Download from Kaggle](https://www.kaggle.com/datasets/computerscience3/public-requirementspure-dataset)
   - **Official Source**: [TU Ilmenau PURE Dataset](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
   - **Size**: ~15 GB
   - **Description**: 10 subjects with RGB video and ground truth heart rate

2. **UBFC-rPPG Dataset**
   - **Official Page**: [UBFC-PHYS](https://sites.google.com/view/ybenezeth/ubfc-phys)
   - **Alternative**: [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
   - **Size**: ~8 GB
   - **Description**: 42 subjects with varying lighting conditions

#### 📁 Dataset Setup Instructions

After downloading the datasets:

```bash
# 1. Create data directory structure
mkdir -p data/PURE
mkdir -p data/UBFC

# 2. Extract datasets to respective folders
# PURE: Extract to data/PURE/
# UBFC: Extract to data/UBFC/

# 3. Verify dataset structure
python scripts/verify_layout.py

# 4. (Optional) If using Google Drive for dataset storage
python scripts/make_symlinks.py
```

#### Expected Directory Structure
```
data/
├── PURE/
│   ├── 01-01/
│   │   ├── 01-01.json           # Ground truth
│   │   └── 01-01/               # Image sequence
│   ├── 01-02/
│   └── ...
└── UBFC/
    ├── subject1/
    │   ├── vid.avi              # Video file
    │   └── ground_truth.txt     # HR ground truth
    ├── subject2/
    └── ...
```

#### 🔗 Alternative: Google Drive Setup

For team collaboration or cloud storage:

```bash
# Set environment variable for your Google Drive path
export GDRIVE_RPPG="G:/My Drive/rppg_datasets"  # Windows
# OR
export GDRIVE_RPPG="~/Google Drive/rppg_datasets"  # macOS/Linux

# Create symbolic links
python scripts/make_symlinks.py
```

#### Evaluate on Datasets
```bash
# Evaluate on PURE dataset
python run_pure_evaluation.py

# Evaluate on UBFC dataset
python run_evaluation.py

# Combined evaluation (both datasets)
python run_combined_evaluation.py

# Incremental improvement evaluation
python run_incremental_evaluation.py
```

## 📊 Performance & Validation

### Current Best Performance

**Model**: Iteration 2 (Haar Cascade + Multi-ROI)
**Dataset**: PURE (24 subjects)

| Metric | Value | Status |
|--------|-------|--------|
| **MAE** | **4.51 BPM** | ✅ Competitive |
| **RMSE** | **6.79 BPM** | ✅ Good |
| **Correlation** | **0.365** | ⚠️ Moderate |
| **Within ±10 BPM** | **87.0%** | ✅ Clinically acceptable |

**See [EVALUATION_RESULTS.md](EVALUATION_RESULTS.md) for detailed analysis**

### Key Findings

✅ **What Works**:
- Multi-ROI averaging (forehead + cheeks) reduces noise
- Simple methods outperform over-engineered filtering
- 33% of subjects achieve excellent MAE < 2 BPM

❌ **Limitations**:
- High subject variability (some fail completely)
- Requires controlled conditions (frontal face, good lighting)
- Classical algorithms plateauing around 4-5 BPM MAE

📖 **Full iteration history**: See [ITERATIONS.md](ITERATIONS.md)

### Supported Datasets
- **PURE**: 24 subjects, controlled laboratory conditions (primary)
- **UBFC-rPPG**: 42 subjects, various lighting conditions
- **Custom Videos**: Any MP4/AVI face video

### Testing Suite
```bash
# Run latest iteration evaluations
python run_iteration4_evaluation.py
python run_iteration4a_evaluation.py  # Illumination only
python run_iteration4b_evaluation.py  # Temporal only
python run_iteration4c_evaluation.py  # Motion only

# Best baseline (Iteration 2)
python run_iteration2_evaluation.py
```

## 🏗️ Architecture

```
📁 Project Structure
├── 🎯 scripts/
│   ├── advanced_vitals_ui.py      # Main monitoring interface
│   ├── vitals_dashboard.py        # Dashboard with graphs
│   ├── evaluate_ubfc.py          # Dataset evaluation
│   ├── simple_rppg_ui.py         # Basic HR monitoring
│   └── run_unsupervised_pos.py   # POS algorithm core
├── 🧪 tests/                      # Comprehensive test suite
├── ⚙️ configs/                    # Configuration files
├── 📊 data/                       # Dataset symlinks
└── 📚 docs/                       # Documentation
```

## 🎛️ Controls & Usage

### Keyboard Controls
- **Q** - Quit application
- **S** - Save vital signs data
- **R** - Reset all measurements
- **SPACE** - Take screenshot
- **G** - Toggle graphs (dashboard mode)
- **P** - Pause/Resume monitoring

### Real-time Monitoring
1. Position face clearly in camera view
2. Ensure stable lighting conditions
3. Wait 10-15 seconds for initialization
4. Monitor color-coded health indicators
5. Save data when needed

## 🔬 Research Applications

### Academic Use Cases
- **Capstone Projects** - Complete rPPG implementation
- **Research Validation** - Algorithm comparison and benchmarking
- **Healthcare Studies** - Remote monitoring research
- **Computer Vision** - Face-based signal extraction

### Algorithm Details
- **POS Algorithm** - Plane-Orthogonal-to-Skin for HR extraction
- **Multi-ROI Processing** - Forehead (HR), cheeks (RR)
- **Signal Filtering** - Butterworth bandpass filters
- **Frequency Analysis** - FFT-based parameter estimation

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- New vital sign detection algorithms
- Enhanced UI/UX design
- Additional dataset support
- Performance optimizations
- Documentation improvements

## 📚 Citation & References

If you use this system in research, please cite:
```bibtex
@misc{facial_vital_signs_2024,
  title={Facial Vital Signs Monitoring System},
  author={[Your Name]},
  year={2024},
  url={https://github.com/carcinojane/facial-vital-sign}
}
```

### Key References
- POS Algorithm: Wang et al., "Algorithmic Principles of Remote PPG" (IEEE TPAMI)
- rPPG Overview: Rouast et al., "Remote heart rate measurement using low-cost RGB face video"

## 📄 License

This project is open source and available for academic research and educational use.

## ❓ Support

- 📖 Check the [comprehensive documentation](REPOSITORY_DESCRIPTION.md)
- 🐛 Report issues via GitHub Issues
- 💡 Request features via GitHub Discussions
- 📧 Contact for research collaboration

---

**🚀 Start monitoring vital signs with just a camera!**

```bash
git clone https://github.com/carcinojane/facial-vital-sign.git && cd facial-vital-sign && python test_enhanced_ui.py
```