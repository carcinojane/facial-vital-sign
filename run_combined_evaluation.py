"""
Combined evaluation script for both PURE and UBFC-rPPG datasets
Generates a unified results.txt file with both datasets
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import time
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from scripts.simple_rppg_ui import RPPGProcessor

# ==================== PURE Dataset Functions ====================

def load_pure_ground_truth(subject_path):
    """Load ground truth HR from PURE subject JSON file"""
    json_files = list(subject_path.glob("*.json"))
    if not json_files:
        return None, None

    json_file = json_files[0]
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        hr_values = []
        timestamps = []

        if '/FullPackage' in data:
            for entry in data['/FullPackage']:
                if 'Value' in entry and 'pulseRate' in entry['Value']:
                    hr = entry['Value']['pulseRate']
                    if hr > 0:
                        hr_values.append(hr)
                        if 'Timestamp' in entry:
                            timestamps.append(entry['Timestamp'])

        if len(hr_values) == 0:
            return None, None

        return np.array(hr_values), np.array(timestamps)

    except Exception as e:
        print(f"Error loading PURE ground truth: {e}")
        return None, None

def process_pure_images(subject_path, subject_name, fps=30, frame_skip=2):
    """Process PURE images with frame skipping"""
    image_dir = subject_path / subject_name
    if not image_dir.exists():
        return None, None

    image_files = sorted(image_dir.glob("*.png"))
    if not image_files:
        return None, None

    image_files = image_files[::frame_skip]
    print(f"  Processing {len(image_files)} frames (skip={frame_skip})")

    processor = RPPGProcessor(fps=fps//frame_skip)
    hr_predictions = []
    timestamps = []

    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        hr, _, _ = processor.process_frame(frame)
        current_time = (idx * frame_skip) / fps
        hr_predictions.append(hr if hr > 0 else np.nan)
        timestamps.append(current_time)

    return np.array(hr_predictions), np.array(timestamps)

# ==================== UBFC Dataset Functions ====================

def load_ubfc_ground_truth(subject_path):
    """Load ground truth HR from UBFC ground_truth.txt"""
    gt_file = subject_path / "ground_truth.txt"
    if not gt_file.exists():
        return None

    try:
        with open(gt_file, 'r') as f:
            lines = f.readlines()

        if len(lines) >= 2:
            hr_line = lines[1].strip()
        else:
            hr_line = lines[0].strip()

        hr_values = [float(x) for x in hr_line.split()]
        return np.array(hr_values)

    except Exception as e:
        print(f"Error loading UBFC ground truth: {e}")
        return None

def process_ubfc_video(video_path, fps=30):
    """Process UBFC video file"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Processing {frame_count} frames at {video_fps:.1f} FPS")

    processor = RPPGProcessor(fps=int(video_fps))
    hr_predictions = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hr, _, _ = processor.process_frame(frame)
        current_time = frame_idx / video_fps
        hr_predictions.append(hr if hr > 0 else np.nan)
        timestamps.append(current_time)
        frame_idx += 1

    cap.release()
    return np.array(hr_predictions), np.array(timestamps)

# ==================== Common Functions ====================

def align_and_calculate_metrics(predictions, pred_ts, ground_truth, gt_ts=None, gt_fps=30):
    """Align predictions with ground truth and calculate metrics"""
    # Create GT timestamps if not provided
    if gt_ts is None:
        gt_ts = np.arange(len(ground_truth)) / gt_fps
    else:
        # Normalize timestamps (PURE uses nanoseconds)
        if gt_ts[0] > 1e12:
            gt_ts = (gt_ts - gt_ts[0]) / 1e9

    # Find overlapping time range
    start_t = max(pred_ts[0], gt_ts[0])
    end_t = min(pred_ts[-1], gt_ts[-1])

    if start_t >= end_t:
        return {}

    # Create common time grid
    common_time = np.arange(start_t, end_t, 1.0)

    # Interpolate
    pred_interp = np.interp(common_time, pred_ts, predictions)
    gt_interp = np.interp(common_time, gt_ts, ground_truth)
    valid_mask = ~(np.isnan(pred_interp) | np.isnan(gt_interp))

    aligned_pred = pred_interp[valid_mask]
    aligned_gt = gt_interp[valid_mask]

    if len(aligned_pred) == 0:
        return {}

    # Calculate metrics
    mae = mean_absolute_error(aligned_gt, aligned_pred)
    rmse = np.sqrt(mean_squared_error(aligned_gt, aligned_pred))
    mape = np.mean(np.abs((aligned_pred - aligned_gt) / aligned_gt)) * 100

    if len(aligned_pred) > 1 and np.std(aligned_pred) > 0 and np.std(aligned_gt) > 0:
        correlation, p_value = pearsonr(aligned_pred, aligned_gt)
    else:
        correlation, p_value = 0, 1

    diff_hr = aligned_pred - aligned_gt
    within_5bpm = np.mean(np.abs(diff_hr) <= 5) * 100
    within_10bpm = np.mean(np.abs(diff_hr) <= 10) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'within_5bpm': within_5bpm,
        'within_10bpm': within_10bpm,
        'num_samples': len(aligned_pred)
    }

# ==================== Main Evaluation ====================

def evaluate_pure(dataset_root):
    """Evaluate PURE dataset"""
    pure_path = Path(dataset_root) / "PURE"
    subject_dirs = sorted([d for d in pure_path.iterdir()
                          if d.is_dir() and '-' in d.name and len(d.name) == 5])

    print(f"\n{'='*70}")
    print(f"EVALUATING PURE DATASET ({len(subject_dirs)} subjects)")
    print(f"{'='*70}")

    results = []

    for i, subject_dir in enumerate(subject_dirs):
        subject_name = subject_dir.name
        print(f"[{i+1}/{len(subject_dirs)}] {subject_name}...", end=' ')

        try:
            gt_hr, gt_ts = load_pure_ground_truth(subject_dir)
            if gt_hr is None:
                print("SKIP (no GT)")
                continue

            start_time = time.time()
            pred_hr, pred_ts = process_pure_images(subject_dir, subject_name, fps=30, frame_skip=2)

            if pred_hr is None or len(pred_hr) == 0:
                print("SKIP (no predictions)")
                continue

            metrics = align_and_calculate_metrics(pred_hr, pred_ts, gt_hr, gt_ts)

            if not metrics:
                print("SKIP (no alignment)")
                continue

            metrics['subject'] = subject_name
            metrics['dataset'] = 'PURE'
            metrics['processing_time'] = time.time() - start_time

            results.append(metrics)
            print(f"MAE={metrics['mae']:.2f}, r={metrics['correlation']:.3f}")

        except Exception as e:
            print(f"ERROR: {e}")

    return results

def evaluate_ubfc(dataset_root):
    """Evaluate UBFC-rPPG dataset"""
    ubfc_path = Path(dataset_root) / "UBFC"
    subject_dirs = sorted([d for d in ubfc_path.iterdir()
                          if d.is_dir() and d.name.startswith('subject')])

    print(f"\n{'='*70}")
    print(f"EVALUATING UBFC-rPPG DATASET ({len(subject_dirs)} subjects)")
    print(f"{'='*70}")

    results = []

    for i, subject_dir in enumerate(subject_dirs):
        subject_name = subject_dir.name
        print(f"[{i+1}/{len(subject_dirs)}] {subject_name}...", end=' ')

        try:
            gt_hr = load_ubfc_ground_truth(subject_dir)
            if gt_hr is None:
                print("SKIP (no GT)")
                continue

            video_files = list(subject_dir.glob("*.avi"))
            if not video_files:
                print("SKIP (no video)")
                continue

            start_time = time.time()
            pred_hr, pred_ts = process_ubfc_video(video_files[0])

            if pred_hr is None or len(pred_hr) == 0:
                print("SKIP (no predictions)")
                continue

            metrics = align_and_calculate_metrics(pred_hr, pred_ts, gt_hr, gt_fps=30)

            if not metrics:
                print("SKIP (no alignment)")
                continue

            metrics['subject'] = subject_name
            metrics['dataset'] = 'UBFC'
            metrics['processing_time'] = time.time() - start_time

            results.append(metrics)
            print(f"MAE={metrics['mae']:.2f}, r={metrics['correlation']:.3f}")

        except Exception as e:
            print(f"ERROR: {e}")

    return results

def generate_combined_report(pure_results, ubfc_results):
    """Generate combined evaluation report"""
    all_results = pure_results + ubfc_results

    if not all_results:
        return "No results to report"

    pure_df = pd.DataFrame(pure_results) if pure_results else pd.DataFrame()
    ubfc_df = pd.DataFrame(ubfc_results) if ubfc_results else pd.DataFrame()
    combined_df = pd.DataFrame(all_results)

    report = []
    report.append("=" * 70)
    report.append("COMBINED rPPG EVALUATION REPORT")
    report.append("=" * 70)
    report.append(f"Algorithm: POS (Plane-Orthogonal-to-Skin)")
    report.append(f"Total subjects evaluated: {len(all_results)}")
    report.append(f"  - PURE: {len(pure_results)} subjects")
    report.append(f"  - UBFC-rPPG: {len(ubfc_results)} subjects")
    report.append("")

    # Overall combined performance
    report.append("OVERALL COMBINED PERFORMANCE:")
    report.append("-" * 40)
    report.append(f"MAE            : {combined_df['mae'].mean():.3f} ± {combined_df['mae'].std():.3f} BPM")
    report.append(f"RMSE           : {combined_df['rmse'].mean():.3f} ± {combined_df['rmse'].std():.3f} BPM")
    report.append(f"MAPE           : {combined_df['mape'].mean():.2f}% ± {combined_df['mape'].std():.2f}%")
    report.append(f"CORRELATION    : {combined_df['correlation'].mean():.3f} ± {combined_df['correlation'].std():.3f}")
    report.append(f"Within 5 BPM   : {combined_df['within_5bpm'].mean():.1f}% ± {combined_df['within_5bpm'].std():.1f}%")
    report.append(f"Within 10 BPM  : {combined_df['within_10bpm'].mean():.1f}% ± {combined_df['within_10bpm'].std():.1f}%")
    report.append("")

    # PURE dataset performance
    if len(pure_df) > 0:
        report.append("PURE DATASET PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"Subjects: {len(pure_results)}")
        report.append(f"MAE            : {pure_df['mae'].mean():.3f} ± {pure_df['mae'].std():.3f} BPM")
        report.append(f"RMSE           : {pure_df['rmse'].mean():.3f} ± {pure_df['rmse'].std():.3f} BPM")
        report.append(f"MAPE           : {pure_df['mape'].mean():.2f}% ± {pure_df['mape'].std():.2f}%")
        report.append(f"CORRELATION    : {pure_df['correlation'].mean():.3f} ± {pure_df['correlation'].std():.3f}")
        report.append(f"Within 5 BPM   : {pure_df['within_5bpm'].mean():.1f}% ± {pure_df['within_5bpm'].std():.1f}%")
        report.append(f"Within 10 BPM  : {pure_df['within_10bpm'].mean():.1f}% ± {pure_df['within_10bpm'].std():.1f}%")
        report.append("")

    # UBFC dataset performance
    if len(ubfc_df) > 0:
        report.append("UBFC-rPPG DATASET PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"Subjects: {len(ubfc_results)}")
        report.append(f"MAE            : {ubfc_df['mae'].mean():.3f} ± {ubfc_df['mae'].std():.3f} BPM")
        report.append(f"RMSE           : {ubfc_df['rmse'].mean():.3f} ± {ubfc_df['rmse'].std():.3f} BPM")
        report.append(f"MAPE           : {ubfc_df['mape'].mean():.2f}% ± {ubfc_df['mape'].std():.2f}%")
        report.append(f"CORRELATION    : {ubfc_df['correlation'].mean():.3f} ± {ubfc_df['correlation'].std():.3f}")
        report.append(f"Within 5 BPM   : {ubfc_df['within_5bpm'].mean():.1f}% ± {ubfc_df['within_5bpm'].std():.1f}%")
        report.append(f"Within 10 BPM  : {ubfc_df['within_10bpm'].mean():.1f}% ± {ubfc_df['within_10bpm'].std():.1f}%")
        report.append("")

    # Subject-by-subject results
    report.append("SUBJECT-BY-SUBJECT RESULTS:")
    report.append("-" * 70)
    report.append(f"{'Dataset':<8} {'Subject':<12} {'MAE':<8} {'RMSE':<8} {'MAPE%':<8} {'r':<8}")
    report.append("-" * 70)

    for result in all_results:
        report.append(f"{result['dataset']:<8} "
                     f"{result['subject']:<12} "
                     f"{result['mae']:<8.2f} "
                     f"{result['rmse']:<8.2f} "
                     f"{result['mape']:<8.1f} "
                     f"{result['correlation']:<8.3f}")

    report.append("=" * 70)

    return "\n".join(report)

def main():
    dataset_root = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\data"

    print("\n" + "=" * 70)
    print("COMBINED RPPG EVALUATION: PURE + UBFC-rPPG")
    print("=" * 70)

    # Evaluate both datasets
    pure_results = evaluate_pure(dataset_root)
    ubfc_results = evaluate_ubfc(dataset_root)

    # Generate combined report
    report = generate_combined_report(pure_results, ubfc_results)
    print("\n" + report)

    # Save results
    with open('results.txt', 'w') as f:
        f.write(report)

    timestamp = int(time.time())
    all_results = pure_results + ubfc_results
    if all_results:
        pd.DataFrame(all_results).to_csv(f'combined_results_{timestamp}.csv', index=False)

    print(f"\n{'='*70}")
    print("Evaluation complete!")
    print(f"Report saved to: results.txt")
    print(f"Detailed results: combined_results_{timestamp}.csv")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
