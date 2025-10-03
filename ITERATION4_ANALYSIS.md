# Iteration 4: Signal Quality Enhancements - Analysis Report

## Executive Summary

**Verdict: ❌ NO IMPROVEMENT - Revert to Iteration 2**

Iteration 4 introduced signal quality enhancements (illumination normalization, temporal consistency, motion detection) but resulted in **worse overall performance** compared to Iteration 2 baseline.

- **MAE increased** by 0.89 BPM (19.6% worse)
- **RMSE increased** by 0.42 BPM (6.2% worse)
- **Correlation improved** by 0.04 (11.0% better) - only positive metric
- **Within 5 BPM accuracy dropped** from 80.6% to 73.2%

## Changes Introduced in Iteration 4

### 1. Illumination Normalization
- Green channel normalization to reduce lighting variations
- Applied per-frame with exponential smoothing

### 2. Temporal Consistency Filtering
- Frame-to-frame signal difference tracking
- Reject frames with excessive temporal variations
- Threshold: mean + 2×std of differences

### 3. Motion Detection & Rejection
- Face landmark stability analysis
- Track centroid movement between frames
- Reject frames exceeding motion threshold (5 pixels)

## Performance Results

### Overall Metrics

| Metric | Iteration 2 | Iteration 4 | Change | % Change |
|--------|-------------|-------------|--------|----------|
| MAE | 4.51 BPM | 5.40 BPM | +0.89 | +19.6% ❌ |
| RMSE | 6.79 BPM | 7.22 BPM | +0.42 | +6.2% ❌ |
| Correlation | 0.365 | 0.405 | +0.04 | +11.0% ✅ |
| Within 5 BPM | 80.6% | 73.2% | -7.4% | -9.2% ❌ |
| Within 10 BPM | 87.0% | 83.5% | -3.4% | -3.9% ❌ |

### Subject-Level Analysis

**Performance Breakdown:**
- 7 subjects improved (29.2%)
- 8 subjects worsened (33.3%)
- 9 subjects similar (37.5%)

**Biggest Improvements:**
- 04-03: MAE 6.48 → 1.64 (-4.84 BPM) ✅
- 02-02: MAE 13.23 → 9.90 (-3.32 BPM) ✅
- 04-02: MAE 9.28 → 7.51 (-1.77 BPM) ✅

**Biggest Regressions:**
- 02-05: MAE 6.82 → 16.00 (+9.18 BPM) ❌
- 10-06: MAE 16.06 → 22.87 (+6.81 BPM) ❌
- 10-05: MAE 9.52 → 15.96 (+6.44 BPM) ❌
- 02-06: MAE 2.88 → 9.21 (+6.32 BPM) ❌

## Root Cause Analysis

### Why Did Signal Quality Enhancements Fail?

#### 1. **Over-Aggressive Frame Rejection**
The temporal consistency and motion detection filters likely rejected too many valid frames:
- Subjects with natural movements (02-05, 02-06, 10-05, 10-06) showed massive regressions
- Reduced effective signal length → worse frequency domain analysis
- Lost critical pulse information during rejection

#### 2. **Illumination Normalization Side Effects**
Green channel normalization may have:
- Removed important DC component variations that carry pulse information
- Introduced artifacts in already well-lit videos
- Over-smoothed signals, reducing pulse amplitude

#### 3. **Poor Generalization**
The filters worked well for some subjects (04-03, 02-02) but failed catastrophically for others:
- Suggests the thresholds are not universally applicable
- May need adaptive thresholds based on video characteristics
- One-size-fits-all approach doesn't work for diverse recording conditions

#### 4. **Correlation Improvement Paradox**
Correlation improved (+11%) while MAE worsened (+19.6%):
- Suggests the signal shape/pattern tracking improved
- BUT absolute accuracy degraded significantly
- Likely due to systematic bias introduced by normalization

## Lessons Learned

### What Worked
1. **Some challenging cases improved** (04-03: -4.84 BPM improvement)
2. **Correlation tracking** showed the approach has potential for signal pattern matching
3. **Framework is sound** - the quality enhancement pipeline can be refined

### What Failed
1. **Static thresholds** don't generalize across subjects
2. **Frame rejection** is too aggressive and loses critical data
3. **Illumination normalization** may remove signal along with noise
4. **No fallback mechanism** when quality filters fail

### Key Insights
- **Simple is better**: Iteration 2's straightforward multi-ROI approach outperforms complex filtering
- **Don't fix what isn't broken**: Adding complexity without clear need hurts performance
- **Validate incrementally**: Should have tested each enhancement separately before combining
- **Subject diversity matters**: What helps one subject can catastrophically hurt another

## Recommendations

### Immediate Action
**REVERT to Iteration 2** as the current best-performing system.

### Future Iteration Ideas (If Pursuing Quality Enhancements)

#### Option A: Adaptive Thresholds
- Make motion/temporal thresholds subject-specific
- Calibrate using initial N frames
- Allow wider tolerance ranges

#### Option B: Selective Application
- Only apply quality filters when confidence is low
- Use quality metrics to decide whether to filter or not
- Fallback to raw signals if filtering degrades results

#### Option C: Incremental Testing
- Test illumination normalization alone (Iteration 4a)
- Test temporal filtering alone (Iteration 4b)
- Test motion rejection alone (Iteration 4c)
- Only combine if each individual component helps

#### Option D: Different Direction Entirely
- Focus on post-processing instead of pre-filtering
- Improve frequency domain analysis (better FFT windows, harmonic analysis)
- Explore ensemble methods (combine predictions from multiple ROIs differently)
- Try alternative rPPG algorithms (CHROM, ICA-based)

## Technical Details

### Code Changes
See `scripts/simple_rppg_ui.py` lines ~240-290 for signal quality enhancement implementation:
- `normalize_illumination()`: Green channel normalization
- `check_temporal_consistency()`: Frame-to-frame difference filtering
- `detect_motion()`: Landmark centroid stability checking

### Evaluation Framework
Script: `run_iteration4_evaluation.py`
- Compares Iteration 2 (enhancements disabled) vs Iteration 4 (enhancements enabled)
- Uses flag-based toggling for A/B comparison
- Processes all 24 UBFC-rPPG subjects

## Conclusion

Iteration 4's signal quality enhancements, while conceptually sound, degraded performance in practice. The MAE increased by 19.6%, making this iteration objectively worse than the Iteration 2 baseline.

**Key Takeaway**: Complex filtering can introduce more problems than it solves. Iteration 2 remains the current best approach.

---

**Date:** 2025-10-03
**Evaluation Dataset:** UBFC-rPPG (24 subjects)
**Result Files:**
- Comparison: `iteration4_comparison.txt`
- Detailed CSV: `iteration4_results_1759469724.csv`
