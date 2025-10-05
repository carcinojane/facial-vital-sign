# Performance Gap Analysis: Why MAE is Still High

**Date**: 2025-10-03
**Current Best**: MAE 4.51 BPM (Haar + Multi-ROI)
**Target**: MAE 2-3 BPM (Published POS benchmarks)
**Gap**: ~2x worse than state-of-the-art

---

## 📊 Current Performance Assessment

### What We've Achieved
- ✅ Baseline established: 4.57 BPM
- ✅ Multi-ROI improvement: -1.3% MAE, -15% RMSE, +13% correlation
- ✅ Systematic methodology validated
- ✅ Face detection evaluated (not limiting factor)

### What We Haven't Addressed
- ❌ Illumination normalization
- ❌ Motion compensation
- ❌ Advanced signal processing
- ❌ Temporal filtering
- ❌ Parameter optimization

---

## 🔍 Root Cause Analysis: Why MAE is 4.51 Instead of 2.5

### 1. **No Illumination Normalization** (LIKELY HIGH IMPACT)

**Current Implementation**:
```python
# We just extract raw RGB values
mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)
```

**What's Missing**:
- No compensation for ambient light changes
- No skin tone normalization
- No local contrast enhancement

**Published Methods Do This**:
```python
# Normalized RGB per ROI
r_norm = R / (R + G + B + epsilon)
g_norm = G / (R + G + B + epsilon)
b_norm = B / (R + G + B + epsilon)
```

**Expected Impact**: 20-30% improvement (0.9-1.4 BPM reduction)

---

### 2. **Suboptimal Bandpass Filter Parameters** (MODERATE IMPACT)

**Current Settings**:
```python
lowcut = 0.7 Hz   # 42 BPM
highcut = 3.0 Hz  # 180 BPM
```

**Issues**:
- Very wide range (42-180 BPM) lets through lots of noise
- Most people at rest: 60-100 BPM (1.0-1.67 Hz)
- We're including unnecessary frequency bands

**Better Approach**:
```python
# Adaptive based on estimated HR
lowcut = estimated_hr - 20 BPM
highcut = estimated_hr + 20 BPM
# Iteratively refine
```

**Expected Impact**: 10-15% improvement (0.45-0.68 BPM reduction)

---

### 3. **No Temporal Smoothing** (MODERATE IMPACT)

**Current Implementation**:
- Each 10-second window processed independently
- No continuity between estimates
- Allows wild frame-to-frame jumps

**What's Missing**:
```python
# Kalman filter for temporal smoothing
# Assumes HR changes gradually
predicted_hr = kalman.predict(previous_hr)
updated_hr = kalman.update(current_measurement)
```

**Published Methods**:
- Kalman filtering
- Moving average smoothing
- HR trajectory constraints

**Expected Impact**: 15-25% improvement (0.68-1.13 BPM reduction)

---

### 4. **No Motion Artifact Rejection** (MODERATE-HIGH IMPACT)

**Current Implementation**:
- Process every window regardless of quality
- No detection of motion artifacts
- No rejection of corrupted segments

**What's Missing**:
```python
# Motion detection (optical flow, frame differencing)
if motion_detected > threshold:
    skip_window()  # Don't include in evaluation
```

**PURE Dataset**:
- Subjects sometimes move slightly
- Blinks, micro-expressions cause artifacts
- We're including these corrupted windows in MAE calculation

**Expected Impact**: 10-20% improvement (0.45-0.90 BPM reduction)

---

### 5. **Simple POS Implementation** (MODERATE IMPACT)

**Our POS Implementation**:
```python
# Simplified version
C1 = normalized[:, 0] - normalized[:, 1]  # R - G
C2 = normalized[:, 0] + normalized[:, 1] - 2 * normalized[:, 2]  # R + G - 2B
alpha = std(C1) / std(C2)
hr_signal = C1 - alpha * C2
```

**Potential Issues**:
- No signal quality assessment before projection
- Fixed alpha calculation (could be adaptive)
- No verification that projection is optimal

**Published POS**:
- More sophisticated normalization
- Adaptive projection parameters
- Signal quality metrics guide processing

**Expected Impact**: 5-10% improvement (0.23-0.45 BPM reduction)

---

### 6. **Peak Detection in Frequency Domain** (LOW-MODERATE IMPACT)

**Current Approach**:
```python
# Find peak in FFT
dominant_freq = freqs[np.argmax(np.abs(fft_values[valid_idx]))]
hr = dominant_freq * 60
```

**Issues**:
- Noise can create spurious peaks
- No verification that peak is physiologically plausible
- No use of harmonics to verify HR

**Better Approach**:
```python
# Multi-harmonic verification
fundamental = find_peak(fft)
if peak_exists_at(2 * fundamental) and peak_exists_at(3 * fundamental):
    hr = fundamental  # Verified by harmonics
else:
    reject_window()  # Likely noise
```

**Expected Impact**: 5-10% improvement (0.23-0.45 BPM reduction)

---

### 7. **No Per-Subject Calibration** (LOW IMPACT for general use)

**Current Approach**:
- Same parameters for all subjects
- No learning from initial frames

**Possible Enhancement**:
```python
# Use first 30 seconds to estimate subject-specific parameters
baseline_hr = estimate_initial_hr()
skin_tone = estimate_skin_reflectance()
optimal_roi = find_best_roi(baseline_signal_quality)
```

**Expected Impact**: 5-10% improvement (0.23-0.45 BPM reduction)
**Caveat**: Makes system less general-purpose

---

## 📈 Cumulative Impact Estimate

If we implemented ALL the missing components:

| Component | Expected MAE Reduction | Cumulative MAE |
|-----------|----------------------|----------------|
| **Current** | - | **4.51 BPM** |
| + Illumination normalization | -1.1 BPM (25%) | 3.41 BPM |
| + Temporal smoothing (Kalman) | -0.68 BPM (20%) | 2.73 BPM |
| + Motion artifact rejection | -0.55 BPM (20%) | 2.18 BPM |
| + Adaptive bandpass filter | -0.33 BPM (15%) | 1.85 BPM |
| + Improved POS implementation | -0.14 BPM (7.5%) | **1.71 BPM** |

**Note**: These are rough estimates assuming independence, which isn't true. Actual combined effect might be less due to interactions.

**Realistic Target with All Improvements**: **2.0-2.5 BPM MAE**

---

## 🎯 Prioritized Improvement Roadmap

### **HIGH PRIORITY** (Biggest Bang for Buck)

#### 1. Illumination Normalization (Expected: -25%)
**Effort**: Low (1-2 hours)
**Impact**: High

```python
def normalize_illumination(roi):
    """Chrominance-based normalization"""
    # Convert to normalized rgb
    rgb_sum = roi[:, :, 0] + roi[:, :, 1] + roi[:, :, 2] + 1e-6
    r_norm = roi[:, :, 0] / rgb_sum
    g_norm = roi[:, :, 1] / rgb_sum
    b_norm = roi[:, :, 2] / rgb_sum
    return np.stack([r_norm, g_norm, b_norm], axis=2)
```

#### 2. Temporal Smoothing (Expected: -20%)
**Effort**: Moderate (3-4 hours)
**Impact**: High

```python
from scipy.signal import savgol_filter

# Apply Savitzky-Golay filter to HR estimates
smoothed_hr = savgol_filter(hr_estimates, window_length=5, polyorder=2)
```

Or implement simple Kalman filter for HR tracking.

#### 3. Motion Artifact Detection (Expected: -20%)
**Effort**: Moderate (2-3 hours)
**Impact**: High

```python
def detect_motion(current_frame, previous_frame):
    """Simple frame differencing for motion detection"""
    diff = np.abs(current_frame - previous_frame).mean()
    return diff > threshold  # Skip window if too much motion
```

---

### **MEDIUM PRIORITY** (Moderate Gains)

#### 4. Adaptive Bandpass Filtering (Expected: -15%)
**Effort**: Moderate (2-3 hours)
**Impact**: Moderate

```python
# Two-pass approach
# Pass 1: Wide band to get rough HR estimate
rough_hr = estimate_hr(signal, lowcut=0.7, highcut=3.0)

# Pass 2: Narrow band around rough estimate
refined_hr = estimate_hr(signal,
                         lowcut=rough_hr - 0.3,  # ±18 BPM
                         highcut=rough_hr + 0.3)
```

#### 5. Signal Quality Assessment (Expected: -10%)
**Effort**: Low-Moderate (2 hours)
**Impact**: Moderate

```python
def calculate_snr(signal, hr_freq):
    """Signal-to-Noise Ratio at HR frequency"""
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fps)

    # Power at HR frequency
    signal_power = np.abs(fft[np.argmin(np.abs(freqs - hr_freq))])**2

    # Total noise power (excluding HR and harmonics)
    noise_power = np.sum(np.abs(fft)**2) - signal_power

    return signal_power / noise_power

# Reject windows with low SNR
if calculate_snr(signal, estimated_hr) < snr_threshold:
    skip_window()
```

---

### **LOW PRIORITY** (Diminishing Returns)

#### 6. Better Peak Detection
**Effort**: Low (1 hour)
**Impact**: Low

#### 7. Per-Subject Calibration
**Effort**: High (5-6 hours)
**Impact**: Low-Moderate (and reduces generalizability)

---

## 🚀 Recommended Next Iteration (Iteration 4)

### **Quick Wins Strategy**

Implement the top 3 high-priority items in a single iteration:

**Iteration 4: Signal Quality Enhancements**
1. Illumination normalization (chrominance)
2. Simple temporal smoothing (moving average or Savitzky-Golay)
3. Basic motion detection (frame differencing)

**Expected Combined Impact**: 40-50% reduction
**Estimated Result**: **MAE ~2.5-2.7 BPM** (from 4.51)

**Time Investment**: 6-9 hours
**Risk**: Low (all are well-established techniques)

---

## 📊 Realistic Performance Expectations

### After Iteration 4 (Signal Quality)
- **Expected MAE**: 2.5-2.7 BPM
- **Expected Correlation**: 0.50-0.60
- **Gap to SOTA**: Still ~0.5 BPM behind, but respectable

### After Further Optimization (Iterations 5-6)
- **Potential MAE**: 2.0-2.3 BPM
- **Potential Correlation**: 0.65-0.75
- **Gap to SOTA**: Competitive for undergraduate capstone

### State-of-the-Art (Published Methods)
- **MAE**: 1.5-2.0 BPM
- **Correlation**: 0.85-0.95
- **Requires**: Years of research, deep learning, extensive optimization

---

## 💡 Key Insights

### 1. We've Picked the "Low-Hanging Fruit"
- Face detection: ✓ Evaluated (not limiting factor)
- Multi-ROI: ✓ Implemented (15% RMSE improvement)
- Bug fixes: ✓ Resolved (FPS mismatch)

**Remaining improvements require more sophisticated signal processing.**

### 2. Current Performance is Actually Reasonable
For a capstone project:
- ✅ Systematic methodology
- ✅ Reproducible results
- ✅ Well-documented iterations
- ✅ Evidence-based decisions
- ⚠️ Accuracy still below SOTA but understandable for scope

### 3. The Gap is Explainable
**What Published Methods Have That We Don't**:
1. Illumination normalization
2. Advanced temporal filtering
3. Motion artifact rejection
4. Extensive parameter tuning
5. Years of optimization
6. Larger validation datasets

**Our 4.51 BPM is reasonable given these omissions.**

---

## 🎓 For Your Capstone Report

### How to Frame Current Performance

**✅ GOOD Framing**:

> "The implemented system achieved MAE of 4.51 BPM on the PURE dataset, representing a 15% RMSE improvement over baseline through Multi-ROI spatial averaging. While this performance is approximately 2x higher error than published POS implementations (2-3 BPM), gap analysis identifies specific missing components: (1) illumination normalization, (2) temporal smoothing, and (3) motion artifact rejection. These omissions are documented with expected impact estimates, providing a clear roadmap for future iterations. The systematic methodology and comprehensive documentation demonstrate engineering rigor appropriate for an undergraduate capstone project."

**❌ BAD Framing**:

> "We achieved 4.51 BPM which isn't as good as published methods."

### Discussion Points

**Strengths**:
1. Systematic improvement methodology validated
2. Clear understanding of current limitations
3. Prioritized roadmap for future work based on impact analysis
4. Honest reporting of both positive and negative results

**Limitations** (be upfront):
1. Performance gap exists (~2x vs published benchmarks)
2. Missing illumination normalization (high impact)
3. No temporal filtering (high impact)
4. Limited to controlled conditions (PURE dataset only)

**Future Work** (specific and actionable):
1. Implement top 3 signal quality enhancements (Iteration 4)
2. Expected to close gap to 2.5-2.7 BPM (~45% improvement)
3. UBFC validation for realistic conditions
4. Consider deep learning approaches for further gains

---

## ✅ Honest Assessment

### Is 4.51 BPM "Good"?

**For Clinical Use**: ❌ No (too high error)
**For Wellness Apps**: ⚠️ Marginal (acceptable with disclaimers)
**For Research Prototype**: ✅ Yes (demonstrates feasibility)
**For Undergraduate Capstone**: ✅ Yes (with proper context and roadmap)

### What Would Make It "Great"?

**Minimum Acceptable** (for capstone): 3.5-4.0 BPM
**Good** (competitive): 2.5-3.0 BPM
**Excellent** (SOTA): 1.5-2.0 BPM

**Your Current 4.51 BPM**: At the "acceptable" threshold. Implementing Iteration 4 (signal quality) would push you into "good" territory.

---

## 🎯 Decision Point: What to Do Next?

### Option 1: **Stop Here, Write Report**
**Pros**:
- You have complete Iterations 0-3 with full documentation
- Methodology is solid and well-validated
- Negative results (Iteration 3) add value
- Can frame 4.51 BPM as reasonable for scope

**Cons**:
- Performance gap is large (~2x)
- Clearly identified improvements left on table
- Reviewers might ask "why didn't you try illumination normalization?"

### Option 2: **One More Iteration (Recommended)**
**Implement Iteration 4: Signal Quality Enhancements**
- Illumination normalization (2 hours)
- Temporal smoothing (2 hours)
- Motion detection (2 hours)
- Evaluation (1.5 hours)
- **Total**: ~7-8 hours

**Expected Outcome**: MAE 2.5-2.7 BPM (nearly competitive)
**Report Impact**: Stronger results, demonstrates full capability

### Option 3: **Multiple Additional Iterations**
**Time**: 15-20 hours
**Risk**: Diminishing returns, time pressure

---

## 📊 Recommendation

**Do ONE more iteration (Iteration 4) focusing on the top 3 high-impact improvements:**

1. ✅ Illumination normalization (2 hours) → -25% expected
2. ✅ Temporal smoothing (2 hours) → -20% expected
3. ✅ Motion detection (2 hours) → -20% expected

**Total Time**: ~8 hours including evaluation
**Expected Result**: **MAE 2.5-2.7 BPM** (from 4.51)
**Report Impact**: Much stronger, competitive performance

**Then STOP and write the report with 4 complete iterations showing progression:**
- Iteration 0: Baseline (4.57 BPM)
- Iteration 2: Multi-ROI (4.51 BPM, -15% RMSE)
- Iteration 3: MediaPipe (4.99 BPM, negative result, valuable insight)
- Iteration 4: Signal Quality (2.5-2.7 BPM, -40% improvement) ← **Strong finish**

This gives you a **complete story** with **competitive results** in **reasonable time**.

---

**Would you like to proceed with Iteration 4 to close the performance gap?**
