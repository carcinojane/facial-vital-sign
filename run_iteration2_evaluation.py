"""
ITERATION 2: Multi-ROI Evaluation
Compares baseline (forehead-only) vs Multi-ROI (forehead + cheeks) performance
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

def process_pure_images(subject_path, subject_name, use_multi_roi, fps=30, frame_skip=2):
    """Process PURE images with specified ROI configuration"""
    image_dir = subject_path / subject_name
    if not image_dir.exists():
        return None, None

    image_files = sorted(image_dir.glob("*.png"))[::frame_skip]
    if not image_files:
        return None, None

    # FIXED: Adjust FPS for frame skipping (effective FPS = original FPS / frame_skip)
    processor = RPPGProcessor(fps=fps//frame_skip, use_multi_roi=use_multi_roi)
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

def evaluate_configuration(dataset_root, use_multi_roi, config_name):
    """Evaluate with specific configuration"""
    pure_path = Path(dataset_root) / "PURE"
    subject_dirs = sorted([d for d in pure_path.iterdir()
                          if d.is_dir() and '-' in d.name and len(d.name) == 5])

    print(f"\n{'='*70}")
    print(f"Evaluating: {config_name}")
    print(f"Multi-ROI: {'ENABLED' if use_multi_roi else 'DISABLED (Baseline)'}")
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

            pred_hr, pred_ts = process_pure_images(subject_dir, subject_name, use_multi_roi, fps=30, frame_skip=2)

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

def generate_comparison_report(baseline_results, multi_roi_results):
    """Generate before/after comparison report"""
    df_baseline = pd.DataFrame(baseline_results)
    df_multi_roi = pd.DataFrame(multi_roi_results)

    report = []
    report.append("=" * 80)
    report.append("ITERATION 2: MULTI-ROI IMPROVEMENT EVALUATION")
    report.append("=" * 80)
    report.append("")
    report.append("COMPARISON: Baseline (Forehead-Only) vs Multi-ROI (Forehead + Cheeks)")
    report.append("")
    report.append("-" * 80)
    report.append("OVERALL PERFORMANCE:")
    report.append("-" * 80)
    report.append("")

    metrics = ['mae', 'rmse', 'correlation', 'within_5bpm', 'within_10bpm']

    report.append(f"{'Metric':<20} {'Baseline':<15} {'Multi-ROI':<15} {'Delta':<15} {'% Change':<10}")
    report.append("-" * 80)

    for metric in metrics:
        baseline_val = df_baseline[metric].mean()
        multi_roi_val = df_multi_roi[metric].mean()
        delta = multi_roi_val - baseline_val
        pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0

        if metric in ['mae', 'rmse', 'mape']:
            # Lower is better
            improvement = "✅" if delta < 0 else "❌"
        else:
            # Higher is better
            improvement = "✅" if delta > 0 else "❌"

        report.append(f"{metric.upper():<20} {baseline_val:<15.3f} {multi_roi_val:<15.3f} "
                     f"{delta:<15.3f} {pct_change:<10.1f}% {improvement}")

    report.append("")
    report.append("-" * 80)
    report.append("SUBJECT-BY-SUBJECT COMPARISON:")
    report.append("-" * 80)
    report.append(f"{'Subject':<12} {'Baseline MAE':<15} {'Multi-ROI MAE':<15} {'Delta':<15} {'Status':<10}")
    report.append("-" * 80)

    improved = 0
    worsened = 0
    unchanged = 0

    for idx, row in df_baseline.iterrows():
        subject = row['subject']
        baseline_mae = row['mae']

        multi_row = df_multi_roi[df_multi_roi['subject'] == subject]
        if len(multi_row) == 0:
            continue

        multi_mae = multi_row.iloc[0]['mae']
        delta = multi_mae - baseline_mae

        if delta < -0.5:
            status = "✅ IMPROVED"
            improved += 1
        elif delta > 0.5:
            status = "❌ WORSENED"
            worsened += 1
        else:
            status = "➖ SIMILAR"
            unchanged += 1

        report.append(f"{subject:<12} {baseline_mae:<15.2f} {multi_mae:<15.2f} {delta:<15.2f} {status:<10}")

    report.append("")
    report.append("-" * 80)
    report.append("SUMMARY:")
    report.append("-" * 80)
    report.append(f"Subjects Improved: {improved}/{len(df_baseline)} ({improved/len(df_baseline)*100:.1f}%)")
    report.append(f"Subjects Worsened: {worsened}/{len(df_baseline)} ({worsened/len(df_baseline)*100:.1f}%)")
    report.append(f"Subjects Unchanged: {unchanged}/{len(df_baseline)} ({unchanged/len(df_baseline)*100:.1f}%)")
    report.append("")

    overall_mae_improvement = df_baseline['mae'].mean() - df_multi_roi['mae'].mean()
    overall_pct = (overall_mae_improvement / df_baseline['mae'].mean()) * 100

    if overall_pct > 5:
        verdict = "✅ SIGNIFICANT IMPROVEMENT - Deploy Multi-ROI"
    elif overall_pct > 0:
        verdict = "⚠️ MINOR IMPROVEMENT - Consider deployment"
    else:
        verdict = "❌ NO IMPROVEMENT - Keep baseline"

    report.append(f"VERDICT: {verdict}")
    report.append(f"Overall MAE Improvement: {overall_mae_improvement:.2f} BPM ({overall_pct:.1f}%)")
    report.append("")
    report.append("=" * 80)

    return "\n".join(report)

def main():
    dataset_root = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\data"

    print("\n" + "=" * 80)
    print("ITERATION 2 EVALUATION: Multi-ROI Comparison")
    print("=" * 80)

    # Evaluate baseline
    print("\nPhase 1: Evaluating Baseline (Forehead-Only)...")
    baseline_results = evaluate_configuration(dataset_root, use_multi_roi=False, config_name="Baseline (Forehead-Only)")

    # Evaluate Multi-ROI
    print("\nPhase 2: Evaluating Multi-ROI (Forehead + Cheeks)...")
    multi_roi_results = evaluate_configuration(dataset_root, use_multi_roi=True, config_name="Multi-ROI (Forehead + Cheeks)")

    # Generate comparison report
    print("\nPhase 3: Generating Comparison Report...")
    report = generate_comparison_report(baseline_results, multi_roi_results)

    # Save results (must save before printing to avoid encoding issues)
    with open('iteration2_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # Try to print report (may fail on Windows console with Unicode)
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\nReport generated (see iteration2_comparison.txt for full details)")

    timestamp = int(time.time())
    df_baseline = pd.DataFrame(baseline_results)
    df_multi_roi = pd.DataFrame(multi_roi_results)

    df_baseline['configuration'] = 'Baseline'
    df_multi_roi['configuration'] = 'Multi-ROI'

    combined = pd.concat([df_baseline, df_multi_roi])
    combined.to_csv(f'iteration2_results_{timestamp}.csv', index=False)

    print(f"\n{'='*80}")
    print("Iteration 2 evaluation complete!")
    print(f"Comparison report: iteration2_comparison.txt")
    print(f"Detailed results: iteration2_results_{timestamp}.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
