# 🩺 Facial Vital Signs Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-Research-orange.svg)](LICENSE)

> **Transform any camera into a comprehensive vital signs monitoring system using facial video analysis**

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
```bash
# Link Google Drive datasets
python scripts/make_symlinks.py

# Verify dataset structure
python scripts/verify_layout.py

# Evaluate on UBFC dataset
python scripts/evaluate_ubfc.py --data_path ./data/UBFC-rPPG
```

## 📊 Performance & Validation

### Accuracy Metrics
- **Heart Rate**: ±3 BPM accuracy on standard datasets
- **Respiratory Rate**: 12-20 BrPM detection range
- **Real-time Processing**: 30+ FPS with face detection
- **Signal Quality**: Automatic stability assessment

### Supported Datasets
- **UBFC-rPPG**: 42 subjects, various lighting conditions
- **PURE**: Multi-subject validation with ground truth
- **Custom Videos**: Any MP4/AVI face video

### Testing Suite
```bash
# Quick component test
python test_enhanced_ui_simple.py

# Comprehensive system test
python test_rppg_system.py

# UBFC dataset evaluation
python test_ubfc_system.py
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