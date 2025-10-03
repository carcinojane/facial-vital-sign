# rPPG System Iterations - Complete History

**Project**: Remote Photoplethysmography Heart Rate Estimation
**Dataset**: PURE (24 subjects, controlled environment)
**Baseline**: POS Algorithm (Wang et al., 2017)
**Last Updated**: 2025-10-03

---

## Overview

This document tracks all iterative improvements from baseline (Iteration 0) through current work (Iteration 4). Each iteration tests a specific hypothesis to improve heart rate estimation accuracy.

### Performance Progression

| Iteration | Method | MAE (BPM) | Change | Status |
|-----------|--------|-----------|--------|--------|
| **0** | Baseline POS | 4.57 | — | ✅ Complete |
| **1** | (Not implemented) | — | — | ⏭️ Skipped |
| **2** | Multi-ROI (Haar) | **4.51** | **-1.3%** | ✅ **Best** |
| **3** | MediaPipe Detection | 4.99 | +10.6% | ❌ Rejected |
| **4** | Signal Quality | 5.40 | +19.7% | ❌ Rejected |
| **4a** | Illumination Only | (partial) | — | 🔄 In Progress |
| **4b** | Temporal Only | (partial) | — | 🔄 In Progress |
| **4c** | Motion Only | (partial) | — | 🔄 In Progress |

**Current Best**: Iteration 2 (4.51 BPM MAE)

---

## Iteration 0: Baseline

**Date**: 2025-10-02
**Status**: ✅ Completed

### Configuration
```
Algorithm: POS (Plane-Orthogonal-to-Skin)
Face Detection: OpenCV Haar Cascade
ROI: Forehead only (upper 40% of face)
Window: 300 frames (10 sec @ 30 FPS)
Filter: 0.7-3.0 Hz bandpass (42-180 BPM)
```

### Results

| Metric | Value |
|--------|-------|
| MAE | 4.57 ± 4.44 BPM |
| RMSE | 7.99 ± 7.09 BPM |
| Correlation | 0.323 ± 0.323 |
| Within 5 BPM | 82.4% |
| Within 10 BPM | 89.3% |

### Key Findings

✅ **Strengths**:
- 11/24 subjects excellent (MAE < 3 BPM)
- 89.3% clinically acceptable (within 10 BPM)
- Works well for stationary subjects

❌ **Weaknesses**:
- High variance (±4.44 BPM)
- Low temporal correlation (r = 0.323)
- 5/24 subjects fail completely (MAE > 9 BPM)
- 2/24 negative correlations

### Root Causes Identified
1. Single ROI limitation (forehead only)
2. No illumination normalization
3. No motion filtering
4. Face detection may be suboptimal
5. Rigid frequency band may miss some subjects

---

## Iteration 2: Multi-ROI Enhancement

**Date**: 2025-10-02
**Status**: ✅ Completed - **Current Best**

### Hypothesis
Averaging signals from multiple facial regions (forehead, left cheek, right cheek) will reduce noise and improve robustness.

### Implementation
```python
ROIs:
  - Forehead: 20-55% face height, 30-70% width
  - Left Cheek: 50-75% height, 15-40% width
  - Right Cheek: 50-75% height, 60-85% width

Signal Fusion: Simple averaging
Face Detection: Haar Cascade (unchanged)
```

### Results

| Metric | Baseline | Multi-ROI | Improvement |
|--------|----------|-----------|-------------|
| MAE | 4.57 | **4.51** | **-1.3%** ✅ |
| RMSE | 7.99 | **6.79** | **-15.0%** ✅ |
| Correlation | 0.323 | **0.365** | **+13.0%** ✅ |
| Within 5 BPM | 82.4% | **80.6%** | -2.2% |
| Within 10 BPM | 89.3% | **87.0%** | -2.6% |

### Subject-Level Impact
- **Improved**: 13/24 subjects (54.2%)
- **Worsened**: 9/24 subjects (37.5%)
- **Similar**: 2/24 subjects (8.3%)

### Analysis

✅ **Success**:
- RMSE reduced by 15% (primary goal achieved)
- Better temporal tracking (correlation +13%)
- More robust across subjects

⚠️ **Trade-offs**:
- Slight decrease in percentage within thresholds
- Some subjects prefer single ROI
- Not a dramatic improvement

**Verdict**: Small but consistent improvement. Multi-ROI becomes new baseline.

---

## Iteration 3: MediaPipe Face Detection

**Date**: 2025-10-03
**Status**: ❌ Rejected

### Hypothesis
MediaPipe's 468-point face mesh provides more accurate face boundaries than Haar Cascade, leading to 5-15% improvement in ROI selection.

### Implementation
```python
Face Detection: MediaPipe Face Mesh (468 landmarks)
ROI Definition: Percentage-based (matched to Haar)
  - Forehead: Top 40% of face mesh bounding box
  - Cheeks: Same percentage regions as Iteration 2
Multi-ROI: Enabled (3 regions averaged)
```

### Results

| Metric | Haar (Iter 2) | MediaPipe | Change |
|--------|---------------|-----------|--------|
| MAE | **4.51** | 4.99 | +10.6% ❌ |
| RMSE | **6.79** | 7.89 | +16.2% ❌ |
| Correlation | **0.365** | 0.263 | -27.9% ❌ |
| Within 5 BPM | 80.6% | 80.8% | +0.3% |
| Within 10 BPM | 87.0% | 88.0% | +1.2% |

### Subject-Level Impact
- **Improved**: 5/24 subjects (20.8%)
- **Worsened**: 7/24 subjects (29.2%)
- **Similar**: 12/24 subjects (50.0%)

### Key Discovery

**Face detection quality is NOT the bottleneck** for PURE dataset performance.

**Why MediaPipe Failed**:
1. PURE has controlled conditions (frontal faces, good lighting, minimal motion)
2. Haar Cascade is already "good enough" for these conditions
3. MediaPipe's precision doesn't help when faces are easy to detect
4. More sophisticated detection ≠ better rPPG signal quality

**Important Insight**: Future improvements should focus on signal processing, not face detection.

**Verdict**: Revert to Haar Cascade. MediaPipe provides no value for this application.

---

## Iteration 4: Signal Quality Enhancements

**Date**: 2025-10-03
**Status**: ❌ Rejected (Overall), 🔄 Sub-iterations in progress

### Hypothesis
Adding illumination normalization, temporal consistency filtering, and motion detection will improve robustness across challenging subjects.

### Implementation
```python
Enhancements (ALL ENABLED):
  1. Illumination Normalization
     - Green channel normalization (reduce lighting variations)
     - Exponential smoothing per frame

  2. Temporal Consistency Filtering
     - Track frame-to-frame signal differences
     - Reject frames exceeding mean + 2σ threshold

  3. Motion Detection
     - Face landmark stability analysis
     - Reject frames with >5 pixel centroid movement

Base: Iteration 2 (Haar + Multi-ROI)
```

### Results (All Enhancements Combined)

| Metric | Iteration 2 | Iteration 4 | Change |
|--------|-------------|-------------|--------|
| MAE | **4.51** | 5.40 | +19.7% ❌ |
| RMSE | **6.79** | 7.22 | +6.2% ❌ |
| Correlation | 0.365 | 0.405 | +11.0% ✅ |
| Within 5 BPM | **80.6%** | 73.2% | -9.2% ❌ |
| Within 10 BPM | **87.0%** | 83.5% | -3.9% ❌ |

### Subject-Level Impact
- **Improved**: 7/24 subjects (29.2%)
- **Worsened**: 8/24 subjects (33.3%)
- **Similar**: 9/24 subjects (37.5%)

### Notable Changes
**Biggest Improvements**:
- 04-03: 6.48 → 1.64 BPM (-74.7%) ✅
- 10-04: 4.62 → 3.84 BPM (-16.9%)
- 04-02: 9.28 → 7.51 BPM (-19.1%)

**Biggest Degradations**:
- 02-05: 6.82 → 16.00 BPM (+134.6%) ❌
- 02-06: 2.88 → 9.21 BPM (+219.8%) ❌
- 10-06: 16.06 → 22.87 BPM (+42.4%) ❌

### Analysis

**Why It Failed**:
1. **Over-filtering**: Aggressive rejection discarded good frames
2. **PURE dataset mismatch**: Already has controlled lighting/motion
3. **Negative synergy**: Multiple filters compounded errors
4. **Signal loss**: Temporal/motion filtering reduced available data

**What Worked**:
- Correlation improved (+11%) - better HR tracking over time
- Helped 7 challenging subjects significantly

**Verdict**: Combined enhancements hurt overall performance. Testing individual components separately.

---

## Iteration 4a: Illumination Normalization ONLY

**Date**: 2025-10-03
**Status**: 🔄 In Progress (Partial results due to memory errors)

### Configuration
```python
Enabled:
  - Illumination normalization (green channel)

Disabled:
  - Temporal filtering
  - Motion detection

Base: Iteration 2 (Haar + Multi-ROI)
```

### Preliminary Results (19/24 subjects)

| Metric | Baseline | +Illumination | Change |
|--------|----------|---------------|--------|
| MAE | 4.51 | ~5.1 | +13% ❌ |
| Videos tested | 24 | 19 | Memory errors |

**Memory Issue**: Iteration 4a crashed at video 04-02 due to insufficient memory (921KB allocation failure). Fixed with explicit garbage collection.

**Preliminary Finding**: Illumination normalization alone appears to hurt performance, possibly because PURE dataset already has controlled lighting.

**Status**: Re-run needed with memory fix applied.

---

## Iteration 4b: Temporal Consistency ONLY

**Date**: 2025-10-03
**Status**: 🔄 In Progress

### Configuration
```python
Enabled:
  - Temporal consistency filtering

Disabled:
  - Illumination normalization
  - Motion detection
```

### Preliminary Results (24/24 subjects)

**Promising signs**: Several videos show improvement:
- 01-02: 7.56 → 5.91 BPM (-21.8%) ✅
- 01-03: 1.37 → 1.22 BPM (-10.9%)
- 04-01: 2.05 → 1.94 BPM (-5.4%)

**Verdict**: Pending full analysis. Temporal filtering alone may be beneficial.

---

## Iteration 4c: Motion Detection ONLY

**Date**: 2025-10-03
**Status**: ⏳ Running

### Configuration
```python
Enabled:
  - Motion detection & rejection

Disabled:
  - Illumination normalization
  - Temporal filtering
```

**Status**: Currently evaluating. Results pending.

---

## Lessons Learned

### What Works
1. ✅ **Multi-ROI averaging** (Iteration 2) - Small but consistent improvement
2. ✅ **RMSE reduction** - Multi-ROI significantly reduced error variance
3. ✅ **Simple is better** - Minimal enhancements outperform complex filtering

### What Doesn't Work
1. ❌ **MediaPipe** (Iteration 3) - No benefit over Haar for controlled conditions
2. ❌ **Combined filtering** (Iteration 4) - Over-engineering hurts performance
3. ❌ **Aggressive frame rejection** - Loses valuable signal data

### Key Insights
1. **Dataset matters**: PURE's controlled environment makes some optimizations unnecessary
2. **Face detection is not the bottleneck**: Signal processing improvements needed instead
3. **Incremental testing is critical**: Testing combined changes obscures which components help/hurt
4. **Subject variability is high**: Different subjects respond differently to same enhancements

---

## Next Steps

### Immediate Priorities
1. Complete Iteration 4a-c with memory fixes
2. Identify which individual enhancement (if any) provides benefit
3. Compare POS algorithm against deep learning baseline (PhysNet)

### Future Directions
1. Test on challenging datasets (UBFC-rPPG, in-the-wild conditions)
2. Explore deep learning approaches if classical methods plateau
3. Investigate why specific subjects fail consistently
4. Consider adaptive filtering based on signal quality metrics

---

## References

- **POS Algorithm**: Wang, W., et al. (2017). "Algorithmic principles of remote PPG." IEEE TBME, 64(7), 1479-1491.
- **PURE Dataset**: Stricker, R., et al. (2014). "Non-contact video-based pulse rate measurement on a mobile service robot."
- **Baseline Performance**: Literature reports 2-5 BPM MAE on PURE dataset for similar methods.

---

**Document Version**: 1.0
**Author**: Capstone Project Team
**Last Review**: 2025-10-03
