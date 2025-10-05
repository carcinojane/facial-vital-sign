"""
ITERATION 4b: Temporal Consistency Filtering ONLY
Tests temporal consistency filtering enhancement in isolation
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

def process_pure_images(subject_path, subject_name, use_temporal, fps=30, frame_skip=2):
    """Process PURE images with temporal consistency filtering only"""
    image_dir = subject_path / subject_name
    if not image_dir.exists():
        return None, None

    image_files = sorted(image_dir.glob("*.png"))[::frame_skip]
    if not image_files:
        return None, None

    # ITERATION 4b: Test temporal consistency filtering ONLY
    processor = RPPGProcessor(
        fps=fps//frame_skip,
        use_multi_roi=True,
        use_mediapipe=False,
        use_illumination_norm=False,  # DISABLED
        use_temporal_filter=use_temporal,  # TEST THIS
        use_motion_detection=False  # DISABLED
    )
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

def evaluate_configuration(dataset_root, use_temporal, config_name):
    """Evaluate with specific configuration"""
    pure_path = Path(dataset_root) / "PURE"
    subject_dirs = sorted([d for d in pure_path.iterdir()
                          if d.is_dir() and '-' in d.name and len(d.name) == 5])

    print(f"\n{'='*70}")
    print(f"Evaluating: {config_name}")
    print(f"Temporal Consistency Filtering: {'ENABLED' if use_temporal else 'DISABLED'}")
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

            pred_hr, pred_ts = process_pure_images(subject_dir, subject_name, use_temporal, fps=30, frame_skip=2)

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

def generate_comparison_report(baseline_results, enhanced_results):
    """Generate before/after comparison report"""
    df_baseline = pd.DataFrame(baseline_results)
    df_enhanced = pd.DataFrame(enhanced_results)

    report = []
    report.append("=" * 80)
    report.append("ITERATION 4b: TEMPORAL CONSISTENCY FILTERING EVALUATION")
    report.append("=" * 80)
    report.append("")
    report.append("COMPARISON: Iteration 2 Baseline vs 4b (+ Temporal Consistency ONLY)")
    report.append("")
    report.append("-" * 80)
    report.append("OVERALL PERFORMANCE:")
    report.append("-" * 80)
    report.append("")

    metrics = ['mae', 'rmse', 'correlation', 'within_5bpm', 'within_10bpm']

    report.append(f"{'Metric':<20} {'Baseline':<15} {'4b (Temporal)':<15} {'Delta':<15} {'% Change':<10}")
    report.append("-" * 80)

    for metric in metrics:
        baseline_val = df_baseline[metric].mean()
        enhanced_val = df_enhanced[metric].mean()
        delta = enhanced_val - baseline_val
        pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0

        if metric in ['mae', 'rmse', 'mape']:
            status = "✅" if delta < 0 else "❌"
        else:
            status = "✅" if delta > 0 else "❌"

        report.append(f"{metric.upper():<20} {baseline_val:<15.3f} {enhanced_val:<15.3f} "
                     f"{delta:<15.3f} {pct_change:<10.1f}% {status}")

    report.append("")
    report.append("-" * 80)
    report.append("SUBJECT-BY-SUBJECT COMPARISON:")
    report.append("-" * 80)
    report.append(f"{'Subject':<12} {'Baseline MAE':<12} {'4b MAE':<12} {'Delta':<15} {'Status':<10}")
    report.append("-" * 80)

    for _, row_baseline in df_baseline.iterrows():
        subject = row_baseline['subject']
        row_enhanced = df_enhanced[df_enhanced['subject'] == subject]

        if len(row_enhanced) > 0:
            baseline_mae = row_baseline['mae']
            enhanced_mae = row_enhanced.iloc[0]['mae']
            delta = enhanced_mae - baseline_mae

            if abs(delta) < 0.5:
                status = "➖ SIMILAR"
            elif delta < 0:
                status = "✅ IMPROVED"
            else:
                status = "❌ WORSENED"

            report.append(f"{subject:<12} {baseline_mae:<12.2f} {enhanced_mae:<12.2f} {delta:<15.2f} {status:<10}")

    # Summary statistics
    improved = sum(1 for _, row in df_baseline.iterrows()
                   if len(df_enhanced[df_enhanced['subject'] == row['subject']]) > 0
                   and df_enhanced[df_enhanced['subject'] == row['subject']].iloc[0]['mae'] < row['mae'] - 0.5)
    worsened = sum(1 for _, row in df_baseline.iterrows()
                   if len(df_enhanced[df_enhanced['subject'] == row['subject']]) > 0
                   and df_enhanced[df_enhanced['subject'] == row['subject']].iloc[0]['mae'] > row['mae'] + 0.5)
    total = len(df_baseline)
    unchanged = total - improved - worsened

    report.append("")
    report.append("-" * 80)
    report.append("SUMMARY:")
    report.append("-" * 80)
    report.append(f"Subjects Improved: {improved}/{total} ({improved/total*100:.1f}%)")
    report.append(f"Subjects Worsened: {worsened}/{total} ({worsened/total*100:.1f}%)")
    report.append(f"Subjects Unchanged: {unchanged}/{total} ({unchanged/total*100:.1f}%)")
    report.append("")

    overall_improvement = (df_baseline['mae'].mean() - df_enhanced['mae'].mean()) / df_baseline['mae'].mean() * 100
    if overall_improvement > 5:
        verdict = "✅ BENEFICIAL - Include temporal filtering"
    elif overall_improvement > 0:
        verdict = "⚠️ MARGINAL - Consider including"
    else:
        verdict = "❌ HARMFUL - Exclude temporal filtering"

    report.append(f"VERDICT: {verdict}")
    report.append(f"Overall MAE Change: {df_enhanced['mae'].mean() - df_baseline['mae'].mean():.2f} BPM ({overall_improvement:.1f}%)")
    report.append("")
    report.append("=" * 80)

    return "\n".join(report)

def main():
    dataset_root = "C:/Users/janej/OneDrive - National University of Singapore/Capstone Project/rppg-vscode-starter/data"

    print("=" * 80)
    print("ITERATION 4b EVALUATION: Temporal Consistency Filtering ONLY")
    print("=" * 80)

    # Baseline
    print("\nPhase 1: Evaluating Baseline (No Enhancements)...")
    baseline_results = evaluate_configuration(dataset_root, use_temporal=False, config_name="Baseline")

    # With temporal filtering
    print("\nPhase 2: Evaluating with Temporal Consistency Filtering...")
    enhanced_results = evaluate_configuration(dataset_root, use_temporal=True, config_name="4b + Temporal")

    # Generate comparison report
    print("\nPhase 3: Generating Comparison Report...")
    report = generate_comparison_report(baseline_results, enhanced_results)

    # Save results
    with open('iteration4b_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("\nReport generated (see iteration4b_comparison.txt for full details)")

    timestamp = int(time.time())
    df_baseline = pd.DataFrame(baseline_results)
    df_enhanced = pd.DataFrame(enhanced_results)

    df_baseline['configuration'] = 'Baseline'
    df_enhanced['configuration'] = 'Iteration4b_Temporal'

    combined = pd.concat([df_baseline, df_enhanced])
    combined.to_csv(f'iteration4b_results_{timestamp}.csv', index=False)

    print(f"\n{'='*80}")
    print("Iteration 4b evaluation complete!")
    print(f"Comparison report: iteration4b_comparison.txt")
    print(f"Detailed results: iteration4b_results_{timestamp}.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
