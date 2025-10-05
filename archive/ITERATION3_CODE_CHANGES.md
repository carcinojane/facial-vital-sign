# Iteration 3: Code Changes Summary

**Date**: 2025-10-03
**File Modified**: `scripts/simple_rppg_ui.py`
**Lines Changed**: 100-171
**Change Type**: Bug fix - ROI extraction methodology

---

## Overview

Fixed MediaPipe ROI extraction to use percentage-based regions matching Haar Cascade approach, ensuring fair comparison of face detection methods without confounding variables.

---

## Changes Made

### File: `scripts/simple_rppg_ui.py`

#### Function Modified: `_extract_mediapipe_roi(self, frame)`

**Location**: Lines 100-171
**Change Type**: Complete rewrite
**Reason**: Original implementation used inconsistent ROI extraction methodology

---

## BEFORE (Buggy Implementation)

### Approach
Used MediaPipe landmarks to define irregular polygonal ROIs specific to facial anatomy landmarks.

### Code (Lines 100-185 - REMOVED)
```python
def _extract_mediapipe_roi(self, frame):
    """ITERATION 3: Extract ROI using MediaPipe Face Mesh landmarks"""
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.mp_face.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark

    if self.use_multi_roi:
        # ITERATION 3: Precise Multi-ROI using MediaPipe landmarks
        rois = []

        # Forehead region (landmarks 10, 67, 297, 338)
        forehead_points = [10, 67, 109, 10, 338, 297, 332, 338]
        forehead_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                           for i in forehead_points]
        forehead_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(forehead_mask, [np.array(forehead_coords)], 255)
        forehead_roi = frame[forehead_mask > 0]
        if forehead_roi.size > 0:
            # Reshape to maintain spatial structure
            y_coords, x_coords = np.where(forehead_mask > 0)
            if len(y_coords) > 0:
                min_y, max_y = y_coords.min(), y_coords.max()
                min_x, max_x = x_coords.min(), x_coords.max()
                forehead_rect = frame[min_y:max_y+1, min_x:max_x+1]
                rois.append(forehead_rect)

        # Left cheek region (landmarks around left cheek)
        left_cheek_points = [205, 50, 123, 132, 58, 172, 136, 150, 149, 176, 148, 152, 205]
        left_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                       for i in left_cheek_points]
        left_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(left_mask, [np.array(left_coords)], 255)
        y_coords, x_coords = np.where(left_mask > 0)
        if len(y_coords) > 0:
            min_y, max_y = y_coords.min(), y_coords.max()
            min_x, max_x = x_coords.min(), x_coords.max()
            left_cheek_rect = frame[min_y:max_y+1, min_x:max_x+1]
            if left_cheek_rect.size > 0:
                rois.append(left_cheek_rect)

        # Right cheek region (landmarks around right cheek)
        right_cheek_points = [425, 280, 352, 361, 288, 397, 365, 379, 378, 400, 377, 152, 425]
        right_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                        for i in right_cheek_points]
        right_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(right_mask, [np.array(right_coords)], 255)
        y_coords, x_coords = np.where(right_mask > 0)
        if len(y_coords) > 0:
            min_y, max_y = y_coords.min(), y_coords.max()
            min_x, max_x = x_coords.min(), x_coords.max()
            right_cheek_rect = frame[min_y:max_y+1, min_x:max_x+1]
            if right_cheek_rect.size > 0:
                rois.append(right_cheek_rect)

        # Get face bounding box for visualization
        x_min = min([landmarks[i].x for i in range(len(landmarks))]) * w
        x_max = max([landmarks[i].x for i in range(len(landmarks))]) * w
        y_min = min([landmarks[i].y for i in range(len(landmarks))]) * h
        y_max = max([landmarks[i].y for i in range(len(landmarks))]) * h
        face_rect = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

        return rois, face_rect
    else:
        # Single forehead ROI with MediaPipe
        forehead_points = [10, 67, 109, 10, 338, 297, 332, 338]
        forehead_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                           for i in forehead_points]
        forehead_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(forehead_mask, [np.array(forehead_coords)], 255)
        y_coords, x_coords = np.where(forehead_mask > 0)
        if len(y_coords) > 0:
            min_y, max_y = y_coords.min(), y_coords.max()
            min_x, max_x = x_coords.min(), x_coords.max()
            forehead_rect = frame[min_y:max_y+1, min_x:max_x+1]

            # Get face bounding box
            x_min = min([landmarks[i].x for i in range(len(landmarks))]) * w
            x_max = max([landmarks[i].x for i in range(len(landmarks))]) * w
            y_min = min([landmarks[i].y for i in range(len(landmarks))]) * h
            y_max = max([landmarks[i].y for i in range(len(landmarks))]) * h
            face_rect = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

            return [forehead_rect], face_rect

    return None, None
```

### Problems with Original Implementation

1. **Inconsistent ROI Shapes**
   - Haar: Rectangular regions
   - MediaPipe: Irregular polygons defined by landmarks
   - Different shapes → different signal characteristics

2. **Inconsistent ROI Sizes**
   - Haar: Percentage-based (30% of face height for forehead)
   - MediaPipe: Landmark-dependent (varies by face shape/detection)
   - Variable sizes → inconsistent signal quality

3. **Inconsistent Anatomical Coverage**
   - Haar: Fixed percentages cover consistent facial areas
   - MediaPipe: Specific landmarks may map to different anatomical regions
   - Different areas → different blood flow characteristics

4. **Confounded Variables**
   - Testing: Face detection quality (Haar vs MediaPipe)
   - Changed: Face detection + ROI methodology
   - Result: Cannot isolate which factor caused performance change

### Performance Impact
- **MAE**: 4.51 → 5.51 BPM (+22% WORSE)
- **RMSE**: 6.79 → 9.65 BPM (+42% WORSE)
- **Correlation**: 0.365 → 0.215 (-41% WORSE)

---

## AFTER (Fixed Implementation)

### Approach
Use MediaPipe **only for face boundary detection**, then apply **identical percentage-based ROI extraction** as Haar Cascade.

### Code (Lines 100-171 - CURRENT)
```python
def _extract_mediapipe_roi(self, frame):
    """ITERATION 3 (FIXED): Extract ROI using MediaPipe with percentage-based regions

    Uses MediaPipe for accurate face detection, then applies SAME percentage-based
    ROI extraction as Haar Cascade to ensure consistent comparison.
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.mp_face.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark

    # Get face bounding box from all landmarks (more accurate than Haar Cascade)
    all_x = [landmarks[i].x for i in range(len(landmarks))]
    all_y = [landmarks[i].y for i in range(len(landmarks))]

    face_x_min = int(min(all_x) * w)
    face_x_max = int(max(all_x) * w)
    face_y_min = int(min(all_y) * h)
    face_y_max = int(max(all_y) * h)

    face_w = face_x_max - face_x_min
    face_h = face_y_max - face_y_min

    if face_w <= 0 or face_h <= 0:
        return None, None

    if self.use_multi_roi:
        # ITERATION 3 (FIXED): Multi-ROI with SAME percentages as Haar Cascade
        rois = []

        # Forehead: Upper 30% of face, starting 10% from top (matches Haar)
        forehead_y1 = face_y_min + int(0.1 * face_h)
        forehead_y2 = face_y_min + int(0.4 * face_h)
        forehead_roi = frame[forehead_y1:forehead_y2, face_x_min:face_x_max]
        if forehead_roi.size > 0:
            rois.append(forehead_roi)

        # Left cheek: 40-70% vertical, 0-40% horizontal (matches Haar)
        left_y1 = face_y_min + int(0.4 * face_h)
        left_y2 = face_y_min + int(0.7 * face_h)
        left_x1 = face_x_min
        left_x2 = face_x_min + int(0.4 * face_w)
        left_cheek_roi = frame[left_y1:left_y2, left_x1:left_x2]
        if left_cheek_roi.size > 0:
            rois.append(left_cheek_roi)

        # Right cheek: 40-70% vertical, 60-100% horizontal (matches Haar)
        right_y1 = face_y_min + int(0.4 * face_h)
        right_y2 = face_y_min + int(0.7 * face_h)
        right_x1 = face_x_min + int(0.6 * face_w)
        right_x2 = face_x_max
        right_cheek_roi = frame[right_y1:right_y2, right_x1:right_x2]
        if right_cheek_roi.size > 0:
            rois.append(right_cheek_roi)

        face_rect = (face_x_min, face_y_min, face_w, face_h)
        return rois, face_rect
    else:
        # Single forehead ROI: Upper 40% of face (matches Haar baseline)
        forehead_y1 = face_y_min
        forehead_y2 = face_y_min + int(0.4 * face_h)
        forehead_roi = frame[forehead_y1:forehead_y2, face_x_min:face_x_max]

        if forehead_roi.size > 0:
            face_rect = (face_x_min, face_y_min, face_w, face_h)
            return [forehead_roi], face_rect

    return None, None
```

### Advantages of Fixed Implementation

1. **Controlled Variable Testing**
   - Changed: Face detection method (Haar → MediaPipe)
   - Unchanged: ROI extraction methodology (percentage-based)
   - Result: Can isolate face detection quality impact

2. **Consistent ROI Properties**
   - ✅ Same shapes (rectangular)
   - ✅ Same relative sizes (percentages of face)
   - ✅ Same anatomical coverage (forehead/cheeks)
   - ✅ Same signal characteristics

3. **MediaPipe Advantage Preserved**
   - More accurate face boundary (468 landmarks vs simple Haar rectangle)
   - Better handling of face orientation/rotation
   - More robust to partial occlusions
   - Improved face localization in challenging conditions

4. **Fair Comparison**
   - Any performance difference attributable to face detection quality
   - No confounding from ROI methodology changes

### Expected Performance Impact
- **MAE**: 4.51 → 4.10-4.30 BPM (5-10% improvement expected)
- **RMSE**: 6.79 → 6.10-6.45 BPM (5-10% improvement expected)
- **Correlation**: 0.365 → 0.385-0.405 (5-10% improvement expected)

---

## Code Comparison

### Multi-ROI Extraction: Forehead Region

#### BEFORE (Landmark-based)
```python
# Define specific landmarks for forehead polygon
forehead_points = [10, 67, 109, 10, 338, 297, 332, 338]
forehead_coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                   for i in forehead_points]

# Create mask for irregular polygon
forehead_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(forehead_mask, [np.array(forehead_coords)], 255)

# Extract bounding box of polygon
y_coords, x_coords = np.where(forehead_mask > 0)
min_y, max_y = y_coords.min(), y_coords.max()
min_x, max_x = x_coords.min(), x_coords.max()
forehead_rect = frame[min_y:max_y+1, min_x:max_x+1]
```

**Issues**:
- ROI size varies by landmark positions
- ROI shape is irregular polygon
- Different anatomical coverage per subject

#### AFTER (Percentage-based)
```python
# Get face bounding box from all landmarks
face_x_min = int(min(all_x) * w)
face_y_min = int(min(all_y) * h)
face_h = face_y_max - face_y_min

# Apply fixed percentage: upper 30%, starting 10% from top
forehead_y1 = face_y_min + int(0.1 * face_h)
forehead_y2 = face_y_min + int(0.4 * face_h)
forehead_roi = frame[forehead_y1:forehead_y2, face_x_min:face_x_max]
```

**Advantages**:
- ROI size proportional to face (consistent)
- ROI shape is rectangle (matches Haar)
- Same anatomical coverage across subjects

---

## Validation

### How to Verify Fix is Correct

1. **ROI Percentages Match Haar Cascade**
   ```python
   # Haar Cascade (from _extract_haar_roi):
   forehead_y1 = y
   forehead_y2 = y + int(0.4 * h)  # 40% of face height

   # MediaPipe Multi-ROI (from _extract_mediapipe_roi):
   forehead_y1 = face_y_min + int(0.1 * face_h)  # Start 10% from top
   forehead_y2 = face_y_min + int(0.4 * face_h)  # End at 40%
   # Net result: 30% of face height (matches Multi-ROI Haar approach)
   ```

2. **Left Cheek Percentages Match**
   ```python
   # Haar Multi-ROI:
   left_y1 = y + int(0.4 * h)  # 40% from top
   left_y2 = y + int(0.7 * h)  # 70% from top
   left_x1 = x                 # 0% from left
   left_x2 = x + int(0.4 * w)  # 40% from left

   # MediaPipe Multi-ROI (IDENTICAL):
   left_y1 = face_y_min + int(0.4 * face_h)
   left_y2 = face_y_min + int(0.7 * face_h)
   left_x1 = face_x_min
   left_x2 = face_x_min + int(0.4 * face_w)
   ```

3. **Right Cheek Percentages Match**
   ```python
   # Haar Multi-ROI:
   right_x1 = x + int(0.6 * w)  # 60% from left
   right_x2 = x + w             # 100% from left

   # MediaPipe Multi-ROI (IDENTICAL):
   right_x1 = face_x_min + int(0.6 * face_w)
   right_x2 = face_x_max
   ```

### Expected Evaluation Results

**Phase 1: Haar Cascade Baseline**
- Should produce IDENTICAL results to Iteration 2
- MAE: 4.51 BPM (exactly)
- RMSE: 6.79 BPM (exactly)
- Validates evaluation pipeline consistency

**Phase 2: MediaPipe (Fixed)**
- Should show modest improvement (5-15%)
- No dramatic swings in subject-level results
- Consistent direction of change across subjects

---

## Testing Checklist

- [x] MediaPipe installed (Python 3.12)
- [x] All dependencies installed (scikit-learn, pandas, opencv)
- [x] Code compiles without errors
- [x] Percentage-based ROI extraction verified (matches Haar)
- [x] Evaluation script runs successfully
- [ ] Results show improvement (evaluation in progress)
- [ ] Subject-level analysis shows consistency (pending)
- [ ] Documentation updated with final results (pending)

---

## Files Modified Summary

| File | Function | Lines | Change Type |
|------|----------|-------|-------------|
| `scripts/simple_rppg_ui.py` | `_extract_mediapipe_roi()` | 100-171 | Complete rewrite |
| `ITERATION3_CODE_CHANGES.md` | N/A | N/A | Created (this file) |
| `ITERATION3_LESSONS_LEARNED.md` | N/A | N/A | Created |

---

## Diff Summary

**Lines removed**: ~85 (buggy landmark-based implementation)
**Lines added**: ~71 (fixed percentage-based implementation)
**Net change**: -14 lines (simpler, cleaner code)

**Complexity reduction**:
- Before: 3 polygon mask operations + coordinate extraction
- After: Direct percentage-based slicing
- Result: Faster, simpler, more maintainable

---

## Performance Comparison

| Metric | Haar Cascade | MediaPipe (Buggy) | MediaPipe (Fixed) |
|--------|--------------|-------------------|-------------------|
| MAE (BPM) | 4.51 | 5.51 (+22%) | TBD (expected 4.10-4.30) |
| RMSE (BPM) | 6.79 | 9.65 (+42%) | TBD (expected 6.10-6.45) |
| Correlation | 0.365 | 0.215 (-41%) | TBD (expected 0.385-0.405) |
| Implementation | Percentage-based | Landmark polygons | Percentage-based ✓ |

---

**Status**: Re-evaluation in progress
**Expected Completion**: ~60-90 minutes from 18:05 (Oct 3)
**Next Update**: When evaluation completes with final results
