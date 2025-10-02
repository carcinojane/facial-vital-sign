"""
Optimized non-interactive script to run PURE evaluation with incremental saves
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

# Import RPPGProcessor
from scripts.simple_rppg_ui import RPPGProcessor

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
        print(f"Error loading ground truth: {e}")
        return None, None

def process_pure_images_optimized(subject_path, subject_name, fps=30, frame_skip=2):
    """Process PURE images with frame skipping for speed"""
    image_dir = subject_path / subject_name
    if not image_dir.exists():
        return None, None

    image_files = sorted(image_dir.glob("*.png"))
    if not image_files:
        return None, None

    # Skip frames for faster processing
    image_files = image_files[::frame_skip]

    print(f"Processing {subject_name}: {len(image_files)} frames (skipping every {frame_skip})")

    processor = RPPGProcessor(fps=fps//frame_skip)
    hr_predictions = []
    timestamps = []

    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        hr, face_rect, filtered_signal = processor.process_frame(frame)
        current_time = (idx * frame_skip) / fps
        hr_predictions.append(hr if hr > 0 else np.nan)
        timestamps.append(current_time)

    return np.array(hr_predictions), np.array(timestamps)

def calculate_metrics(predictions, ground_truth):
    """Calculate metrics"""
    if len(predictions) == 0 or len(ground_truth) == 0:
        return {}

    mae = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
    mape = np.mean(np.abs((predictions - ground_truth) / ground_truth)) * 100

    if len(predictions) > 1 and np.std(predictions) > 0 and np.std(ground_truth) > 0:
        correlation, p_value = pearsonr(predictions, ground_truth)
    else:
        correlation, p_value = 0, 1

    diff_hr = predictions - ground_truth
    bias = np.mean(diff_hr)
    std_diff = np.std(diff_hr)
    within_5bpm = np.mean(np.abs(diff_hr) <= 5) * 100
    within_10bpm = np.mean(np.abs(diff_hr) <= 10) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'p_value': p_value,
        'bias': bias,
        'std_diff': std_diff,
        'within_5bpm': within_5bpm,
        'within_10bpm': within_10bpm,
        'num_samples': len(predictions)
    }

def main():
    dataset_root = Path(r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\data")
    pure_path = dataset_root / "PURE"

    # Get all subject directories
    subject_dirs = sorted([d for d in pure_path.iterdir()
                          if d.is_dir() and '-' in d.name and len(d.name) == 5])

    print(f"Found {len(subject_dirs)} PURE subjects")
    print("Starting evaluation with optimized processing (frame skip=2)...")

    all_results = []
    successful = 0

    for i, subject_dir in enumerate(subject_dirs):
        subject_name = subject_dir.name
        print(f"\n[{i+1}/{len(subject_dirs)}] Evaluating {subject_name}...")

        try:
            # Load ground truth
            gt_hr, gt_ts = load_pure_ground_truth(subject_dir)
            if gt_hr is None:
                print(f"  Skipping - no ground truth")
                continue

            # Process images with optimization
            start_time = time.time()
            pred_hr, pred_ts = process_pure_images_optimized(subject_dir, subject_name, fps=30, frame_skip=2)

            if pred_hr is None or len(pred_hr) == 0:
                print(f"  Skipping - no predictions")
                continue

            # Normalize GT timestamps
            if gt_ts[0] > 1e12:
                gt_ts = (gt_ts - gt_ts[0]) / 1e9

            # Align predictions with ground truth
            start_t = max(pred_ts[0], gt_ts[0])
            end_t = min(pred_ts[-1], gt_ts[-1])
            common_time = np.arange(start_t, end_t, 1.0)

            pred_interp = np.interp(common_time, pred_ts, pred_hr)
            gt_interp = np.interp(common_time, gt_ts, gt_hr)
            valid_mask = ~(np.isnan(pred_interp) | np.isnan(gt_interp))

            aligned_pred = pred_interp[valid_mask]
            aligned_gt = gt_interp[valid_mask]

            if len(aligned_pred) == 0:
                print(f"  Skipping - no aligned data")
                continue

            # Calculate metrics
            metrics = calculate_metrics(aligned_pred, aligned_gt)
            metrics['subject'] = subject_name
            metrics['processing_time'] = time.time() - start_time

            all_results.append(metrics)
            successful += 1

            print(f"  MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, r={metrics['correlation']:.3f}")

            # Save incrementally every 5 subjects
            if successful % 5 == 0:
                pd.DataFrame(all_results).to_csv('pure_results_incremental.csv', index=False)
                print(f"  [Checkpoint saved: {successful} subjects]")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Generate final report
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)

        report = []
        report.append("=" * 70)
        report.append("PURE DATASET EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"Subjects evaluated: {len(all_results)}")
        report.append(f"Algorithm: POS (Plane-Orthogonal-to-Skin) - Optimized")
        report.append("")
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 40)
        report.append(f"MAE            : {df['mae'].mean():.3f} ± {df['mae'].std():.3f}")
        report.append(f"RMSE           : {df['rmse'].mean():.3f} ± {df['rmse'].std():.3f}")
        report.append(f"MAPE           : {df['mape'].mean():.2f}% ± {df['mape'].std():.2f}%")
        report.append(f"CORRELATION    : {df['correlation'].mean():.3f} ± {df['correlation'].std():.3f}")
        report.append(f"Within 5 Bpm   : {df['within_5bpm'].mean():.1f}% ± {df['within_5bpm'].std():.1f}%")
        report.append(f"Within 10 Bpm  : {df['within_10bpm'].mean():.1f}% ± {df['within_10bpm'].std():.1f}%")
        report.append("")
        report.append("SUBJECT-BY-SUBJECT RESULTS:")
        report.append("-" * 70)
        report.append(f"{'Subject':<12} {'MAE':<8} {'RMSE':<8} {'MAPE%':<8} {'r':<8} {'<5BPM%':<8} {'<10BPM%':<8}")
        report.append("-" * 70)

        for result in all_results:
            report.append(f"{result['subject']:<12} "
                         f"{result['mae']:<8.2f} "
                         f"{result['rmse']:<8.2f} "
                         f"{result['mape']:<8.1f} "
                         f"{result['correlation']:<8.3f} "
                         f"{result['within_5bpm']:<8.1f} "
                         f"{result['within_10bpm']:<8.1f}")

        report_text = "\n".join(report)
        print("\n" + report_text)

        # Save results
        with open('results.txt', 'w') as f:
            f.write(report_text)

        timestamp = int(time.time())
        df.to_csv(f'pure_results_{timestamp}.csv', index=False)

        print(f"\n{'='*70}")
        print("Evaluation complete!")
        print(f"Report saved to: results.txt")
        print(f"Detailed results: pure_results_{timestamp}.csv")
        print(f"{'='*70}")
    else:
        print("No results generated")

if __name__ == "__main__":
    main()
