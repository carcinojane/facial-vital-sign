"""
ITERATION 3: MediaPipe Face Detection Evaluation
Compares Haar Cascade vs MediaPipe face landmarks with Multi-ROI
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

def load_pure_ground_truth(subject_path):
    """Load ground truth HR from PURE JSON"""
    json_files = list(subject_path.glob("*.json"))
    if not json_files:
        return None, None

    try:
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        hr_values, timestamps = [], []
        if '/FullPackage' in data:
            for entry in data['/FullPackage']:
                if 'Value' in entry and 'pulseRate' in entry['Value']:
                    hr = entry['Value']['pulseRate']
                    if hr > 0:
                        hr_values.append(hr)
                        if 'Timestamp' in entry:
                            timestamps.append(entry['Timestamp'])

        return np.array(hr_values) if hr_values else None, np.array(timestamps) if timestamps else None
    except:
        return None, None

def process_pure_images(subject_path, subject_name, use_mediapipe, fps=30, frame_skip=2):
    """Process PURE images with specified face detection method"""
    image_dir = subject_path / subject_name
    if not image_dir.exists():
        return None, None

    image_files = sorted(image_dir.glob("*.png"))[::frame_skip]
    if not image_files:
        return None, None

    # ITERATION 3: Create processor with MediaPipe option
    processor = RPPGProcessor(fps=fps//frame_skip, use_multi_roi=True, use_mediapipe=use_mediapipe)
    hr_predictions, timestamps = [], []

    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        hr, _, _ = processor.process_frame(frame)
        current_time = (idx * frame_skip) / fps
        hr_predictions.append(hr if hr > 0 else np.nan)
        timestamps.append(current_time)

    return np.array(hr_predictions), np.array(timestamps)

def align_and_calculate_metrics(predictions, pred_ts, ground_truth, gt_ts):
    """Align and calculate metrics"""
    if gt_ts[0] > 1e12:
        gt_ts = (gt_ts - gt_ts[0]) / 1e9

    start_t = max(pred_ts[0], gt_ts[0])
    end_t = min(pred_ts[-1], gt_ts[-1])

    if start_t >= end_t:
        return {}

    common_time = np.arange(start_t, end_t, 1.0)
    pred_interp = np.interp(common_time, pred_ts, predictions)
    gt_interp = np.interp(common_time, gt_ts, ground_truth)
    valid_mask = ~(np.isnan(pred_interp) | np.isnan(gt_interp))

    aligned_pred = pred_interp[valid_mask]
    aligned_gt = gt_interp[valid_mask]

    if len(aligned_pred) == 0:
        return {}

    mae = mean_absolute_error(aligned_gt, aligned_pred)
    rmse = np.sqrt(mean_squared_error(aligned_gt, aligned_pred))
    mape = np.mean(np.abs((aligned_pred - aligned_gt) / aligned_gt)) * 100

    if len(aligned_pred) > 1 and np.std(aligned_pred) > 0 and np.std(aligned_gt) > 0:
        correlation, _ = pearsonr(aligned_pred, aligned_gt)
    else:
        correlation = 0

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

def evaluate_configuration(dataset_root, use_mediapipe, config_name):
    """Evaluate with specific configuration"""
    pure_path = Path(dataset_root) / "PURE"
    subject_dirs = sorted([d for d in pure_path.iterdir()
                          if d.is_dir() and '-' in d.name and len(d.name) == 5])

    print(f"\n{'='*70}")
    print(f"Evaluating: {config_name}")
    print(f"Face Detection: {'MediaPipe' if use_mediapipe else 'Haar Cascade'}")
    print(f"{'='*70}")

    results = []

    for i, subject_dir in enumerate(subject_dirs):
        subject_name = subject_dir.name
        print(f"[{i+1}/{len(subject_dirs)}] {subject_name}...", end=' ')

        try:
            gt_hr, gt_ts = load_pure_ground_truth(subject_dir)
            if gt_hr is None:
                print("SKIP")
                continue

            pred_hr, pred_ts = process_pure_images(subject_dir, subject_name, use_mediapipe, fps=30, frame_skip=2)

            if pred_hr is None or len(pred_hr) == 0:
                print("SKIP")
                continue

            metrics = align_and_calculate_metrics(pred_hr, pred_ts, gt_hr, gt_ts)

            if not metrics:
                print("SKIP")
                continue

            metrics['subject'] = subject_name
            results.append(metrics)
            print(f"MAE={metrics['mae']:.2f}, r={metrics['correlation']:.3f}")

        except Exception as e:
            print(f"ERROR: {e}")

    return results

def generate_comparison_report(haar_results, mediapipe_results):
    """Generate before/after comparison report"""
    df_haar = pd.DataFrame(haar_results)
    df_mediapipe = pd.DataFrame(mediapipe_results)

    report = []
    report.append("=" * 80)
    report.append("ITERATION 3: MEDIAPIPE FACE DETECTION EVALUATION")
    report.append("=" * 80)
    report.append("")
    report.append("COMPARISON: Haar Cascade vs MediaPipe Face Landmarks (both with Multi-ROI)")
    report.append("")
    report.append("-" * 80)
    report.append("OVERALL PERFORMANCE:")
    report.append("-" * 80)
    report.append("")

    metrics = ['mae', 'rmse', 'correlation', 'within_5bpm', 'within_10bpm']

    report.append(f"{'Metric':<20} {'Haar Cascade':<15} {'MediaPipe':<15} {'Delta':<15} {'% Change':<10}")
    report.append("-" * 80)

    for metric in metrics:
        haar_val = df_haar[metric].mean()
        mediapipe_val = df_mediapipe[metric].mean()
        delta = mediapipe_val - haar_val
        pct_change = (delta / haar_val * 100) if haar_val != 0 else 0

        if metric in ['mae', 'rmse', 'mape']:
            improvement = "IMPROVED" if delta < 0 else "WORSENED"
        else:
            improvement = "IMPROVED" if delta > 0 else "WORSENED"

        report.append(f"{metric.upper():<20} {haar_val:<15.3f} {mediapipe_val:<15.3f} "
                     f"{delta:<15.3f} {pct_change:<10.1f}% {improvement}")

    report.append("")
    report.append("-" * 80)
    report.append("SUBJECT-BY-SUBJECT COMPARISON:")
    report.append("-" * 80)
    report.append(f"{'Subject':<12} {'Haar MAE':<12} {'MediaPipe MAE':<15} {'Delta':<15} {'Status':<10}")
    report.append("-" * 80)

    for _, row_haar in df_haar.iterrows():
        subject = row_haar['subject']
        row_mp = df_mediapipe[df_mediapipe['subject'] == subject]

        if len(row_mp) > 0:
            haar_mae = row_haar['mae']
            mp_mae = row_mp.iloc[0]['mae']
            delta = mp_mae - haar_mae

            if abs(delta) < 0.5:
                status = "SIMILAR"
            elif delta < 0:
                status = "IMPROVED"
            else:
                status = "WORSENED"

            report.append(f"{subject:<12} {haar_mae:<12.2f} {mp_mae:<15.2f} {delta:<15.2f} {status:<10}")

    # Summary statistics
    improved = sum(1 for _, row in df_haar.iterrows()
                   if len(df_mediapipe[df_mediapipe['subject'] == row['subject']]) > 0
                   and df_mediapipe[df_mediapipe['subject'] == row['subject']].iloc[0]['mae'] < row['mae'] - 0.5)
    worsened = sum(1 for _, row in df_haar.iterrows()
                   if len(df_mediapipe[df_mediapipe['subject'] == row['subject']]) > 0
                   and df_mediapipe[df_mediapipe['subject'] == row['subject']].iloc[0]['mae'] > row['mae'] + 0.5)
    total = len(df_haar)
    unchanged = total - improved - worsened

    report.append("")
    report.append("-" * 80)
    report.append("SUMMARY:")
    report.append("-" * 80)
    report.append(f"Subjects Improved: {improved}/{total} ({improved/total*100:.1f}%)")
    report.append(f"Subjects Worsened: {worsened}/{total} ({worsened/total*100:.1f}%)")
    report.append(f"Subjects Unchanged: {unchanged}/{total} ({unchanged/total*100:.1f}%)")
    report.append("")

    overall_improvement = (df_haar['mae'].mean() - df_mediapipe['mae'].mean()) / df_haar['mae'].mean() * 100
    if overall_improvement > 5:
        verdict = "SIGNIFICANT IMPROVEMENT - Deploy MediaPipe"
    elif overall_improvement > 0:
        verdict = "MINOR IMPROVEMENT - Consider deployment"
    else:
        verdict = "NO IMPROVEMENT - Keep Haar Cascade"

    report.append(f"VERDICT: {verdict}")
    report.append(f"Overall MAE Change: {df_mediapipe['mae'].mean() - df_haar['mae'].mean():.2f} BPM ({overall_improvement:.1f}%)")
    report.append("")
    report.append("=" * 80)

    return "\n".join(report)

def main():
    dataset_root = "C:/Users/janej/OneDrive - National University of Singapore/Capstone Project/rppg-vscode-starter/data"

    print("=" * 80)
    print("ITERATION 3 EVALUATION: MediaPipe Face Detection")
    print("=" * 80)

    # Evaluate Haar Cascade (Iteration 2 baseline)
    print("\nPhase 1: Evaluating Haar Cascade + Multi-ROI (Iteration 2 Baseline)...")
    haar_results = evaluate_configuration(dataset_root, use_mediapipe=False, config_name="Haar + Multi-ROI")

    # Evaluate MediaPipe
    print("\nPhase 2: Evaluating MediaPipe + Multi-ROI...")
    mediapipe_results = evaluate_configuration(dataset_root, use_mediapipe=True, config_name="MediaPipe + Multi-ROI")

    # Generate comparison report
    print("\nPhase 3: Generating Comparison Report...")
    report = generate_comparison_report(haar_results, mediapipe_results)

    # Save results (must save before printing to avoid encoding issues)
    with open('iteration3_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # Try to print report (may fail on Windows console with Unicode)
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\nReport generated (see iteration3_comparison.txt for full details)")

    timestamp = int(time.time())
    df_haar = pd.DataFrame(haar_results)
    df_mediapipe = pd.DataFrame(mediapipe_results)

    df_haar['configuration'] = 'Haar_MultiROI'
    df_mediapipe['configuration'] = 'MediaPipe_MultiROI'

    combined = pd.concat([df_haar, df_mediapipe])
    combined.to_csv(f'iteration3_results_{timestamp}.csv', index=False)

    print(f"\n{'='*80}")
    print("Iteration 3 evaluation complete!")
    print(f"Comparison report: iteration3_comparison.txt")
    print(f"Detailed results: iteration3_results_{timestamp}.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
