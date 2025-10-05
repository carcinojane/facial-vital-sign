# Mobile Phone Video Augmentation Guide

## Overview

This guide explains how to augment laboratory-quality rPPG datasets (PURE, UBFC) to simulate mobile phone camera characteristics. This enables training models for mobile deployment without collecting real mobile phone videos.

## Why Augment for Mobile Quality?

Laboratory datasets (PURE, UBFC) are recorded in controlled conditions with high-quality cameras. Mobile phone videos differ significantly:

| Characteristic | Laboratory Dataset | Mobile Phone Video |
|---------------|-------------------|-------------------|
| **Camera Quality** | High-end research cameras | Consumer smartphone cameras |
| **Compression** | Minimal/lossless | Heavy H.264/H.265 compression |
| **Resolution** | Consistent high resolution | Variable, often lower |
| **Motion** | Tripod-mounted, stable | Handheld, camera shake |
| **Lighting** | Controlled, uniform | Variable, ambient lighting |
| **Noise** | Minimal sensor noise | Higher noise, especially in low light |
| **Color** | Calibrated white balance | Automatic white balance (varies) |

Augmentation bridges this gap by synthetically degrading high-quality videos to match mobile phone characteristics.

## Augmentation System

### Components

1. **`scripts/augment_ubfc_mobile.py`** - Core augmentation engine
2. **`run_augment_ubfc.py`** - Convenient runner script
3. **`PHYSNET_MOBILE_FINETUNING_GUIDE.md`** - Complete fine-tuning workflow

### Mobile Phone Degradations Applied

The augmentation pipeline applies realistic degradations in this order:

#### 1. Resolution Reduction
```python
# Downscale to 50-80% of original resolution
# Then upscale back to original size
# Simulates lower sensor quality
```
- **Light preset**: 0.8x scale (80% quality)
- **Moderate preset**: 0.6x scale (60% quality)
- **Heavy preset**: 0.5x scale (50% quality)

This mimics the effect of lower-quality mobile phone sensors and interpolation artifacts.

#### 2. Motion Blur
```python
# Random direction motion blur
# Probability: 20-40% of frames
# Kernel size: 5-9 pixels
```
Simulates camera shake from handheld recording. Applied randomly to mimic natural hand movement variations.

#### 3. Gaussian Noise
```python
# Sensor noise simulation
# Sigma: 3-8 (depends on preset)
```
Adds realistic sensor noise, especially important for low-light mobile conditions.

#### 4. Brightness Variations
```python
# Random brightness shifts: ±15 to ±35
# Simulates automatic exposure adjustments
```
Mobile phones constantly adjust exposure automatically, creating brightness fluctuations.

#### 5. Color Temperature Shifts
```python
# Warm/cool white balance variations
# Random temperature shift: 0.9-1.1x
```
Simulates automatic white balance adjustments in mobile cameras.

#### 6. Compression Artifacts
```python
# JPEG compression (simulates H.264 video codec)
# Quality: 50-85% (depends on preset)
```
Final step applies compression artifacts similar to H.264/H.265 video encoding.

### Augmentation Presets

Three intensity levels are available:

#### Light Preset
**Use case**: Modern flagship smartphones in good conditions
```yaml
compression_quality: 85
resolution_scale: 0.8
motion_blur_prob: 0.2
motion_blur_size: 5
noise_sigma: 3
brightness_range: (-15, 15)
```

#### Moderate Preset (Default)
**Use case**: Mid-range smartphones, typical usage
```yaml
compression_quality: 70
resolution_scale: 0.6
motion_blur_prob: 0.3
motion_blur_size: 7
noise_sigma: 5
brightness_range: (-25, 25)
```

#### Heavy Preset
**Use case**: Budget smartphones, challenging conditions
```yaml
compression_quality: 50
resolution_scale: 0.5
motion_blur_prob: 0.4
motion_blur_size: 9
noise_sigma: 8
brightness_range: (-35, 35)
```

## Usage

### Quick Start

```bash
# 1. Preview augmentation effects
python run_augment_ubfc.py --preview

# 2. Test on single subject
python run_augment_ubfc.py --sample

# 3. Process full UBFC dataset (37 subjects)
python run_augment_ubfc.py

# 4. Use different preset
python run_augment_ubfc.py --preset heavy
```

### Detailed Usage

#### Preview Mode
Creates side-by-side comparison of original vs augmented frames:

```bash
python run_augment_ubfc.py --preview
```

Output: `augmentation_preview/` directory with:
- `frame_00_original.jpg`, `frame_00_augmented.jpg`
- `frame_01_original.jpg`, `frame_01_augmented.jpg`
- ... (10 frame pairs by default)

#### Sample Mode
Processes only the first subject for quick testing:

```bash
python run_augment_ubfc.py --sample
```

Output: `data/UBFC_MOBILE/subject1/` with augmented video

**Estimated time**: 1-2 minutes

#### Full Dataset Mode
Processes all 37 UBFC subjects:

```bash
python run_augment_ubfc.py
```

Output: `data/UBFC_MOBILE/` with complete augmented dataset

**Estimated time**: 30-60 minutes (depends on hardware)

#### Custom Presets
```bash
# Light degradation (flagship phones)
python run_augment_ubfc.py --preset light

# Moderate degradation (default)
python run_augment_ubfc.py --preset moderate

# Heavy degradation (budget phones)
python run_augment_ubfc.py --preset heavy
```

### Advanced Usage

Direct script invocation with custom parameters:

```bash
# Custom input/output directories
python scripts/augment_ubfc_mobile.py \
  --input data/UBFC \
  --output data/MY_MOBILE_DATASET \
  --preset moderate

# Preview with specific video
python scripts/augment_ubfc_mobile.py \
  --preview data/UBFC/subject5/vid.avi \
  --preview-output my_preview/
```

## Output Structure

The augmented dataset maintains UBFC-compatible structure:

```
data/UBFC_MOBILE/
├── subject1/
│   ├── vid.avi              # Augmented video
│   └── ground_truth.txt     # Copied from original (unchanged)
├── subject2/
│   ├── vid.avi
│   └── ground_truth.txt
├── subject3/
│   ├── vid.avi
│   └── ground_truth.txt
└── ...
    (37 subjects total)
```

**Important**: Ground truth labels are copied unchanged because:
- Heart rate values don't change due to video degradation
- Only visual quality is affected, not physiological signals
- This allows direct fine-tuning with existing labels

## Performance Characteristics

### Processing Speed

Approximate processing times (on modern hardware):

| Mode | Subjects | Estimated Time |
|------|----------|---------------|
| Preview | 1 video (sample frames) | 10-30 seconds |
| Sample | 1 video (full) | 1-2 minutes |
| Full Dataset | 37 videos | 30-60 minutes |

**Factors affecting speed**:
- CPU speed (augmentation is CPU-bound)
- Disk I/O speed (SSD vs HDD)
- Video resolution and length
- Augmentation preset (heavy is slower)

### Storage Requirements

| Dataset | Size |
|---------|------|
| Original UBFC | ~2-3 GB |
| Augmented UBFC_MOBILE | ~2-3 GB |
| **Total** | ~4-6 GB |

Augmented videos have similar size to originals due to compression.

## Integration with Fine-tuning

After augmentation, use the augmented dataset for fine-tuning:

### Step 1: Update Configuration

Edit `physnet_mobile_finetune.yaml`:

```yaml
TRAIN:
  DATA:
    DATA_PATH: "C:/Users/janej/OneDrive - National University of Singapore/Capstone Project/rppg-vscode-starter/data/UBFC_MOBILE"
    CACHED_PATH: "C:/rppg_cache_mobile"

VALID:
  DATA:
    DATA_PATH: "C:/Users/janej/OneDrive - National University of Singapore/Capstone Project/rppg-vscode-starter/data/UBFC_MOBILE"

TEST:
  DATA:
    DATA_PATH: "C:/Users/janej/OneDrive - National University of Singapore/Capstone Project/rppg-vscode-starter/data/UBFC_MOBILE"
```

### Step 2: Run Fine-tuning

```bash
python run_mobile_finetune.py
```

See `PHYSNET_MOBILE_FINETUNING_GUIDE.md` for complete fine-tuning workflow.

## Validation and Quality Checks

### Visual Inspection

After augmentation, visually verify output quality:

```bash
# Open augmented video
# Windows:
start data/UBFC_MOBILE/subject1/vid.avi

# Check that video:
# - Plays correctly
# - Shows realistic degradation
# - Face is still visible
# - Not overly corrupted
```

### Comparison with Original

```bash
# Preview mode shows side-by-side comparison
python run_augment_ubfc.py --preview

# Check augmentation_preview/ folder
# Verify degradation looks realistic
```

### Expected Characteristics

Augmented videos should exhibit:
- ✓ Visible but not extreme compression artifacts
- ✓ Slight blur/softness compared to original
- ✓ Occasional motion blur (natural looking)
- ✓ Subtle brightness variations across frames
- ✓ Face still clearly detectable
- ✗ Not overly noisy or unreadable

## Troubleshooting

### Issue: Augmentation Too Strong

**Symptom**: Videos look corrupted, faces hard to detect

**Solution**: Use lighter preset
```bash
python run_augment_ubfc.py --preset light
```

### Issue: Augmentation Too Weak

**Symptom**: Augmented videos look identical to originals

**Solution**: Use heavier preset or verify augmentation is actually running
```bash
python run_augment_ubfc.py --preset heavy
```

### Issue: Out of Memory

**Symptom**: Script crashes during processing

**Solution**: Process one subject at a time
```bash
# Use sample mode multiple times with different subjects
# Manually copy/process subjects in batches
```

### Issue: Slow Processing

**Symptom**: Taking much longer than expected

**Solution**:
- Close other applications
- Use SSD instead of HDD
- Process fewer subjects initially
- Consider using lighter preset (processes faster)

### Issue: Ground Truth Missing

**Symptom**: "No valid ground truth found" warnings

**Solution**: Original UBFC dataset must have ground_truth.txt files
```bash
# Verify original dataset
ls data/UBFC/subject1/
# Should show: vid.avi, ground_truth.txt
```

## Best Practices

### 1. Always Preview First

Before processing full dataset:
```bash
python run_augment_ubfc.py --preview
```
Verify degradation level matches your needs.

### 2. Test with Sample

Process single subject before full run:
```bash
python run_augment_ubfc.py --sample
```
Ensure script completes successfully.

### 3. Choose Appropriate Preset

- **Light**: Modern phones, good conditions → closest to original quality
- **Moderate**: Typical mobile usage → recommended starting point
- **Heavy**: Worst-case scenarios, budget phones → maximum robustness

### 4. Keep Original Dataset

Never delete original UBFC dataset:
- Augmented dataset supplements, doesn't replace
- Original needed for comparison
- Can re-augment with different presets

### 5. Verify Output Before Fine-tuning

Before spending hours on fine-tuning:
1. Check a few augmented videos play correctly
2. Verify ground truth files copied
3. Test face detection still works on augmented videos

## Technical Details

### Why This Augmentation Order?

The pipeline applies degradations in a specific order to match real mobile camera processing:

1. **Resolution** - Happens at sensor level
2. **Motion blur** - Occurs during image capture
3. **Noise** - Added by sensor
4. **Brightness/Color** - Applied by ISP (Image Signal Processor)
5. **Compression** - Final step before storage

This order produces more realistic results than random application.

### Randomization

Several augmentations use randomization:
- Motion blur direction (0-180°)
- Brightness shift (within range)
- Color temperature (warm vs cool)
- Motion blur probability (per frame)

This creates natural variation across frames, similar to real mobile videos.

### Preserving Heart Rate Signal

Ground truth labels remain valid because:
- Heart rate is encoded in subtle color changes
- While degradation reduces SNR, signal still present
- PhysNet learns to extract HR from degraded signals
- This is exactly what we want for mobile deployment

## Comparison: Augmented vs Real Mobile

### Advantages of Augmentation

✓ **No data collection**: Use existing datasets
✓ **Controlled degradation**: Consistent, reproducible
✓ **Ground truth**: Already have accurate HR labels
✓ **Fast**: 30-60 minutes vs weeks of collection
✓ **Scalable**: Can augment any dataset

### Limitations of Augmentation

✗ **Not perfect simulation**: Some mobile artifacts hard to replicate
✗ **Single degradation model**: Real phones vary widely
✗ **Missing factors**: Screen glare, app UI, real usage patterns

### Recommended Approach

1. **Start with augmentation**: Quick, gets you 80% there
2. **Fine-tune on augmented data**: Learn mobile-robust features
3. **Test on real mobile videos**: Validate performance
4. **Collect real mobile data if needed**: Fill remaining gaps

## Expected Performance Impact

### Before Augmentation Training
Testing PhysNet (trained on original UBFC) on mobile videos:
- Pearson correlation: ~0.75-0.85
- MAE: ~3-5 Hz

### After Augmentation Training
Testing PhysNet (fine-tuned on UBFC_MOBILE) on mobile videos:
- Pearson correlation: ~0.88-0.95 (target)
- MAE: ~1-2 Hz (target)

**Note**: Actual results depend on mobile video quality and dataset size.

## References

- **rPPG-Toolbox**: https://github.com/ubicomplab/rPPG-Toolbox
- **PhysNet Paper**: "Remote Photoplethysmography Signal Measurement from Facial Videos Using Spatio-Temporal Networks"
- **UBFC-rPPG Dataset**: https://sites.google.com/view/ybenezeth/ubfcrppg

## Summary Checklist

- [ ] Preview augmentation effects
- [ ] Test with sample mode (1 subject)
- [ ] Verify augmented video quality
- [ ] Run full dataset augmentation (37 subjects)
- [ ] Check output directory structure
- [ ] Update physnet_mobile_finetune.yaml paths
- [ ] Proceed to fine-tuning

---

**Created**: 2025-10-04
**Version**: 1.0
**Author**: rPPG Mobile Augmentation System
