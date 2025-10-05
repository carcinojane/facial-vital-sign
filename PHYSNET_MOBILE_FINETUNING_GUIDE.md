# PhysNet Fine-tuning Guide for Mobile Phone Videos

## Overview

This guide explains how to fine-tune the PhysNet deep learning model for mobile phone quality videos. Fine-tuning adapts the pre-trained model (trained on laboratory-quality datasets like PURE/UBFC) to work better with real-world mobile phone videos.

## Why Fine-tune for Mobile Videos?

Mobile phone videos differ from laboratory datasets in several ways:

1. **Video Quality**: Variable compression, bitrate, codec artifacts
2. **Lighting**: Less controlled, more variable ambient conditions
3. **Motion**: More camera shake, head movement (handheld recording)
4. **Face Detection**: Different angles, distances, occlusions
5. **Resolution**: Often lower or inconsistent resolution

Fine-tuning helps the model adapt to these characteristics and improve performance on mobile videos.

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: Multi-core processor (4+ cores recommended)
- RAM: 16 GB
- Storage: 20 GB free space
- GPU: Not required but highly recommended

**Recommended:**
- GPU: NVIDIA GPU with 6+ GB VRAM (RTX 3060 or better)
- RAM: 32 GB
- Storage: 50+ GB SSD

### Software Requirements

- Python 3.10 with conda environment
- rPPG conda environment activated
- CUDA toolkit (if using GPU)

### Dataset Requirements

**Minimum Dataset:**
- 50-100 mobile phone videos
- 10-20 different subjects
- 30-60 seconds per video
- Ground truth heart rate measurements

**Recommended Dataset:**
- 200+ mobile phone videos
- 30+ different subjects
- 60-120 seconds per video
- High-quality ground truth (ECG or pulse oximeter)

## Two Approaches to Mobile Fine-tuning

### Approach 1: Use Augmented Laboratory Data (Recommended for Quick Start)

If you don't have real mobile phone videos yet, you can augment the existing UBFC dataset to simulate mobile phone quality. This approach:
- Uses existing UBFC dataset (37 videos with ground truth)
- Applies realistic degradations (compression, noise, motion blur, etc.)
- Provides immediate training data without collection effort
- Good for initial model development and testing

**Quick Start with Augmentation:**
```bash
# Preview augmentation on single video
python scripts/augment_ubfc_mobile.py --preview data/UBFC/subject1/vid.avi

# Test on one subject
python scripts/augment_ubfc_mobile.py --sample

# Process full UBFC dataset (creates data/UBFC_MOBILE)
python scripts/augment_ubfc_mobile.py

# Or with heavy degradation
python scripts/augment_ubfc_mobile.py --preset heavy
```

Then update `physnet_mobile_finetune.yaml`:
```yaml
TRAIN:
  DATA:
    DATA_PATH: "C:/Users/janej/OneDrive - National University of Singapore/Capstone Project/rppg-vscode-starter/data/UBFC_MOBILE"
```

### Approach 2: Collect Real Mobile Phone Videos

For best real-world performance, collect actual mobile phone videos. See Step 1 below for details.

## Step-by-Step Fine-tuning Process

### Step 1: Prepare Your Mobile Dataset

#### 1.1 Create Sample Structure

```bash
# Create sample dataset structure to understand the format
python scripts/prepare_mobile_dataset.py --sample
```

This creates `data/MOBILE/subject_sample/` with example files and README.

#### 1.2 Organize Your Videos

Organize your mobile phone videos in the following structure:

```
data/MOBILE/
├── subject1/
│   ├── vid.avi (or .mp4)
│   └── ground_truth.txt
├── subject2/
│   ├── vid.avi
│   └── ground_truth.txt
└── ...
```

**Ground Truth Format Options:**

Option 1 - Continuous values (one HR per line):
```
75.2
76.1
74.8
75.5
```

Option 2 - Timestamp, HR pairs:
```
0.0 75.2
0.5 76.1
1.0 74.8
1.5 75.5
```

Option 3 - Single line, space-separated:
```
75.2 76.1 74.8 75.5 ...
```

#### 1.3 Automated Organization

If you have raw videos in a directory, use the preparation script:

```bash
# For .avi files
python scripts/prepare_mobile_dataset.py --input path/to/raw_videos/

# For .mp4 files
python scripts/prepare_mobile_dataset.py --input path/to/raw_videos/ --ext .mp4

# Custom output location
python scripts/prepare_mobile_dataset.py --input raw_videos/ --output data/MY_MOBILE
```

The script will:
- Validate video properties (FPS, resolution, duration)
- Look for corresponding ground truth files
- Copy files to organized structure
- Report issues with any videos

### Step 2: Configure Fine-tuning

#### 2.1 Review Configuration File

Open `physnet_mobile_finetune.yaml` and adjust these key parameters:

**Device Settings:**
```yaml
DEVICE: cuda  # Change to 'cpu' if no GPU
NUM_OF_GPU_TRAIN: 1
```

**Training Parameters:**
```yaml
TRAIN:
  BATCH_SIZE: 4  # Reduce to 2 or 1 if GPU memory limited
  EPOCHS: 30     # Increase if you have more data
  LR: 0.00001    # Learning rate (lower = safer for fine-tuning)
```

**Dataset Paths:**
```yaml
TRAIN:
  DATA:
    DATA_PATH: "path/to/data/MOBILE"  # Update if different
    CACHED_PATH: "C:/rppg_cache_mobile"
    BEGIN: 0.0
    END: 0.7  # 70% for training

VALID:
  DATA:
    BEGIN: 0.7
    END: 0.85  # 15% for validation

TEST:
  DATA:
    BEGIN: 0.85
    END: 1.0  # 15% for testing
```

**Model Resume Path:**
```yaml
MODEL:
  RESUME: "rppg_toolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth"
  # Or use: "rppg_toolbox/final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth"
```

#### 2.2 Optimization Strategies

**Strategy A: Full Fine-tuning (Recommended for 100+ videos)**
- Use default config (all layers trainable)
- Lower learning rate (1e-5)
- More epochs (30-50)

**Strategy B: Last-Layer Fine-tuning (For smaller datasets)**
- Modify trainer to freeze early layers
- Higher learning rate (1e-4)
- Fewer epochs (20-30)

**Strategy C: Gradual Unfreezing**
1. Train last layers only (10 epochs)
2. Unfreeze more layers (10 epochs)
3. Fine-tune all layers (10 epochs)

### Step 3: Run Fine-tuning

```bash
# Activate conda environment
conda activate rppg

# Start fine-tuning
python run_mobile_finetune.py
```

**What to Monitor:**

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should decrease, then plateau
3. **Training Time**: 2-8 hours depending on hardware and dataset size

**Expected Output:**
```
====Training Epoch: 0====
[0,    99] loss: 0.450
[0,   199] loss: 0.380
...
validation loss: 0.350

====Training Epoch: 1====
[1,    99] loss: 0.320
...
validation loss: 0.310
Update best model! Best epoch: 1
```

**Early Stopping:**

The training will automatically save the best model based on validation loss. If validation loss stops improving for several epochs, you can stop training (Ctrl+C).

### Step 4: Evaluate Fine-tuned Model

#### 4.1 Create Evaluation Config

```bash
python scripts/evaluate_mobile_model.py --create-config
```

This creates `physnet_mobile_eval.yaml` for testing the fine-tuned model.

#### 4.2 Run Full Evaluation

```bash
# Using the toolbox
python rppg_toolbox/main.py --config_file physnet_mobile_eval.yaml
```

#### 4.3 Quick Test on Single Video

```bash
python scripts/evaluate_mobile_model.py --video data/MOBILE/subject1/vid.avi
```

#### 4.4 Compare with Baseline

```bash
python scripts/evaluate_mobile_model.py \
  --model runs/mobile_finetune/physnet_mobile_finetuned_best.pth \
  --baseline rppg_toolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth
```

### Step 5: Analyze Results

Check the following outputs:

**Model Checkpoints:**
```
runs/mobile_finetune/
├── physnet_mobile_finetuned_Epoch0.pth
├── physnet_mobile_finetuned_Epoch1.pth
├── ...
└── physnet_mobile_finetuned_best.pth  # Best validation epoch
```

**Logs and Metrics:**
```
runs/mobile_finetune/
├── training_log.txt
└── loss_curves.png  # If PLOT_LOSSES_AND_LR: True
```

**Performance Metrics:**

Compare baseline vs fine-tuned performance:

| Metric | Baseline (PURE) | Fine-tuned (Mobile) | Improvement |
|--------|----------------|---------------------|-------------|
| MAE (Hz) | ~1-3 Hz | ~0.5-1 Hz | Better |
| Pearson | ~0.90-0.95 | ~0.95-0.98 | Better |
| RMSE (Hz) | ~2-5 Hz | ~1-2 Hz | Better |

## Troubleshooting

### Issue: Out of Memory (GPU)

**Solution:**
```yaml
TRAIN:
  BATCH_SIZE: 2  # Or 1 for very limited GPU memory
```

### Issue: Out of Memory (CPU/RAM)

**Solution 1 - Reduce dataset size:**
```yaml
TRAIN:
  DATA:
    END: 0.5  # Use only 50% of training data
```

**Solution 2 - Reduce batch size:**
```yaml
TRAIN:
  BATCH_SIZE: 1
```

### Issue: Training Loss Not Decreasing

**Possible Causes:**
1. Learning rate too high
2. Dataset too small or low quality
3. Incorrect ground truth format

**Solutions:**
```yaml
TRAIN:
  LR: 0.000001  # Reduce learning rate by 10x
  EPOCHS: 50    # Train longer
```

### Issue: Validation Loss Increasing (Overfitting)

**Solutions:**
1. Add data augmentation
2. Reduce model complexity (not recommended)
3. Collect more diverse training data
4. Stop training earlier (use best validation checkpoint)

### Issue: Face Detection Failures

**Solution - Use better face detection:**
```yaml
PREPROCESS:
  CROP_FACE:
    BACKEND: 'MediaPipe'  # Better than 'HC' for varied angles
    LARGE_BOX_COEF: 1.2   # Tighter crop
```

### Issue: Ground Truth Format Errors

**Check:**
1. File encoding (should be UTF-8 or ASCII)
2. Consistent delimiter (space, comma, or tab)
3. Numeric values only (no text headers in data rows)

## Advanced Topics

### Custom Data Loader

For non-UBFC format datasets, create custom loader in:
`rppg_toolbox/dataset/data_loader/MOBILELoader.py`

Base it on `UBFCrPPGLoader.py` structure.

### Layer-wise Fine-tuning

Modify `rppg_toolbox/neural_methods/trainer/PhysnetTrainer.py`:

```python
def __init__(self, config, data_loader):
    # ... existing code ...

    # Freeze early layers
    for name, param in self.model.named_parameters():
        if 'encoder' in name:  # Freeze encoder layers
            param.requires_grad = False

    # Only optimize unfrozen parameters
    self.optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr=config.TRAIN.LR
    )
```

### Data Augmentation

Enable in config:
```yaml
PREPROCESS:
  DATA_AUG: ['Flip', 'Rotate']  # Horizontal flip, small rotations
```

### Custom Loss Functions

Modify `PhysnetTrainer.py` to use combined loss:
```python
loss = 0.5 * neg_pearson_loss + 0.3 * mse_loss + 0.2 * freq_loss
```

## Best Practices

### Data Collection

1. **Diverse Conditions**: Record in various lighting, backgrounds, skin tones
2. **Multiple Devices**: Use different mobile phone models if possible
3. **Realistic Scenarios**: Include typical usage patterns (handheld, various distances)
4. **Quality Ground Truth**: Use medical-grade ECG or pulse oximeter

### Training

1. **Start Small**: Begin with subset of data to verify pipeline
2. **Monitor Closely**: Check training/validation loss every few epochs
3. **Save Frequently**: Keep multiple checkpoints
4. **Validate on Held-out Data**: Never test on training data

### Evaluation

1. **Test on New Subjects**: Evaluate on people not in training set
2. **Cross-validation**: Use k-fold cross-validation for small datasets
3. **Real-world Testing**: Test in actual deployment conditions
4. **Compare Baselines**: Always compare with pre-trained model

## Expected Performance

### Baseline (Pre-trained PURE)
- **PURE Dataset**: Pearson ~0.999, MAE ~0.08 Hz
- **UBFC Dataset**: Pearson ~0.984, MAE ~1.36 Hz
- **Mobile Videos**: Pearson ~0.85-0.92, MAE ~2-4 Hz

### After Fine-tuning (Mobile)
- **Target**: Pearson ~0.93-0.97, MAE ~0.8-2 Hz
- **Realistic**: Pearson ~0.90-0.95, MAE ~1-3 Hz

**Note**: Performance depends heavily on:
- Dataset quality and size
- Video conditions (lighting, motion, resolution)
- Ground truth accuracy

## Summary Checklist

- [ ] Prepare mobile video dataset (50+ videos minimum)
- [ ] Organize in UBFC-compatible structure
- [ ] Validate videos and ground truth files
- [ ] Configure `physnet_mobile_finetune.yaml`
- [ ] Set DEVICE (cuda or cpu)
- [ ] Adjust BATCH_SIZE for your hardware
- [ ] Set appropriate learning rate
- [ ] Run `python run_mobile_finetune.py`
- [ ] Monitor training/validation loss
- [ ] Evaluate fine-tuned model
- [ ] Compare with baseline performance
- [ ] Deploy best model

## Files Created

1. **physnet_mobile_finetune.yaml** - Training configuration
2. **run_mobile_finetune.py** - Fine-tuning runner script
3. **scripts/prepare_mobile_dataset.py** - Dataset preparation tool
4. **scripts/augment_ubfc_mobile.py** - UBFC dataset augmentation tool
5. **scripts/evaluate_mobile_model.py** - Evaluation tool
6. **PHYSNET_MOBILE_FINETUNING_GUIDE.md** - This guide

## Augmentation Details

The `augment_ubfc_mobile.py` script simulates mobile phone camera characteristics:

### Degradations Applied

1. **Compression Artifacts**: H.264 JPEG compression (50-85% quality)
2. **Resolution Reduction**: Downscale to 50-80% then upscale (simulates lower sensor quality)
3. **Motion Blur**: Random direction blur (simulates camera shake)
4. **Gaussian Noise**: Sensor noise (sigma 3-8)
5. **Brightness Variations**: Random ±15 to ±35 brightness shifts
6. **Color Temperature**: White balance variations (warm/cool shifts)

### Augmentation Presets

- **Light**: 85% compression, 0.8x resolution, minimal noise
- **Moderate**: 70% compression, 0.6x resolution, moderate noise (default)
- **Heavy**: 50% compression, 0.5x resolution, high noise

### Usage Examples

```bash
# Preview augmentation effects
python scripts/augment_ubfc_mobile.py --preview data/UBFC/subject1/vid.avi

# Test on one subject (quick verification)
python scripts/augment_ubfc_mobile.py --sample

# Process full dataset with moderate degradation
python scripts/augment_ubfc_mobile.py

# Heavy degradation for worst-case mobile scenarios
python scripts/augment_ubfc_mobile.py --preset heavy --output data/UBFC_MOBILE_HEAVY

# Custom paths
python scripts/augment_ubfc_mobile.py --input data/UBFC --output data/MY_MOBILE_DATA
```

The augmented dataset maintains the same structure as UBFC:
```
data/UBFC_MOBILE/
├── subject1/
│   ├── vid.avi (augmented)
│   └── ground_truth.txt (copied from original)
├── subject2/
│   ├── vid.avi
│   └── ground_truth.txt
└── ...
```

## Support and Resources

- **rPPG-Toolbox Documentation**: https://github.com/ubicomplab/rPPG-Toolbox
- **PhysNet Paper**: "Remote Photoplethysmography Signal Measurement from Facial Videos Using Spatio-Temporal Networks"
- **PURE Dataset**: https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure
- **UBFC-rPPG Dataset**: https://sites.google.com/view/ybenezeth/ubfcrppg

---

**Date**: 2025-01-04
**Version**: 1.0
