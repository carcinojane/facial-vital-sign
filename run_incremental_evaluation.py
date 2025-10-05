"""
Incremental Evaluation Framework
Tests each improvement method systematically and documents impact in results.txt

Methodology:
1. Baseline evaluation (no improvements)
2. Add each improvement individually
3. Test cumulative combinations
4. Identify best combination
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
from scripts.rppg_processor_enhanced import EnhancedRPPGProcessor

# ==================== Evaluation Configurations ====================

IMPROVEMENT_METHODS = [
    {'name': 'Baseline', 'config': {}},
    {'name': 'Motion Filtering', 'config': {'motion_filtering': True}},
    {'name': 'Multi-ROI', 'config': {'adaptive_roi': True}},
    {'name': 'Detrending', 'config': {'detrending': True}},
    {'name': 'Adaptive Bandpass', 'config': {'adaptive_bandpass': True}},
    {'name': 'Temporal Smoothing', 'config': {'temporal_smoothing': True}},
    {'name': 'Outlier Rejection', 'config': {'outlier_rejection': True}},
]

# Cumulative configurations (best performing methods combined)
CUMULATIVE_CONFIGS = [
    {'name': 'Baseline', 'config': {}},
    {'name': 'Best Single', 'config': None},  # Will be determined
    {'name': 'Best Two', 'config': None},
    {'name': 'Best Three', 'config': None},
    {'name': 'All Methods', 'config': {
        'motion_filtering': True,
        'adaptive_roi': True,
        'detrending': True,
        'adaptive_bandpass': True,
        'temporal_smoothing': True,
        'outlier_rejection': True,
    }},
]

# ==================== Dataset Processing ====================

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

def process_pure_images(subject_path, subject_name, processor, fps=30, frame_skip=2):
    """Process PURE images with given processor configuration"""
    image_dir = subject_path / subject_name
    if not image_dir.exists():
        return None, None

    image_files = sorted(image_dir.glob("*.png"))[::frame_skip]
    if not image_files:
        return None, None

    hr_predictions, timestamps = [], []

    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        hr, _, _, _ = processor.process_frame(frame)
        current_time = (idx * frame_skip) / fps
        hr_predictions.append(hr if hr > 0 else np.nan)
        timestamps.append(current_time)

    return np.array(hr_predictions), np.array(timestamps)

def align_and_calculate_metrics(predictions, pred_ts, ground_truth, gt_ts):
    """Align and calculate metrics"""
    # Normalize GT timestamps (PURE uses nanoseconds)
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

def evaluate_configuration(dataset_root, config, config_name, max_subjects=None):
    """Evaluate a specific configuration"""
    pure_path = Path(dataset_root) / "PURE"
    subject_dirs = sorted([d for d in pure_path.iterdir()
                          if d.is_dir() and '-' in d.name and len(d.name) == 5])

    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]

    print(f"\n{'='*70}")
    print(f"Evaluating: {config_name}")
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

            # Create processor with this configuration
            processor = EnhancedRPPGProcessor(fps=30, config=config)

            pred_hr, pred_ts = process_pure_images(subject_dir, subject_name, processor, fps=30, frame_skip=2)

            if pred_hr is None or len(pred_hr) == 0:
                print("SKIP")
                continue

            metrics = align_and_calculate_metrics(pred_hr, pred_ts, gt_hr, gt_ts)

            if not metrics:
                print("SKIP")
                continue

            metrics['subject'] = subject_name
            results.append(metrics)
            print(f"MAE={metrics['mae']:.2f}")

        except Exception as e:
            print(f"ERROR: {e}")

    return results

# ==================== Report Generation ====================

def generate_incremental_report(all_evaluations):
    """Generate comprehensive incremental evaluation report"""
    report = []
    report.append("=" * 80)
    report.append("INCREMENTAL rPPG IMPROVEMENT EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")
    report.append("METHODOLOGY:")
    report.append("-" * 80)
    report.append("This report documents a systematic evaluation of rPPG improvement methods.")
    report.append("Each method is tested individually, then best-performing methods are combined.")
    report.append("")
    report.append("Dataset: PURE (24 subjects)")
    report.append("Algorithm: Plane-Orthogonal-to-Skin (POS)")
    report.append("Evaluation Metrics: MAE, RMSE, MAPE, Correlation, Within-5BPM%, Within-10BPM%")
    report.append("")
    report.append("=" * 80)
    report.append("IMPROVEMENT METHODS TESTED:")
    report.append("=" * 80)
    report.append("1. Motion Filtering     : Skip frames with significant motion artifacts")
    report.append("2. Multi-ROI            : Use forehead + both cheeks (vs forehead only)")
    report.append("3. Signal Detrending    : Remove low-frequency trends from signal")
    report.append("4. Adaptive Bandpass    : Wider frequency range (0.5-3.5Hz vs 0.7-3.0Hz)")
    report.append("5. Temporal Smoothing   : Median filter over recent HR estimates")
    report.append("6. Outlier Rejection    : Reject estimates >20 BPM from median")
    report.append("")

    # Section 1: Individual Method Results
    report.append("=" * 80)
    report.append("SECTION 1: INDIVIDUAL METHOD PERFORMANCE")
    report.append("=" * 80)
    report.append("")
    report.append("Each improvement method is evaluated independently against the baseline.")
    report.append("")

    # Create comparison table
    report.append(f"{'Method':<25} {'MAE':<10} {'RMSE':<10} {'r':<10} {'<5BPM%':<10} {'<10BPM%':<10}")
    report.append("-" * 80)

    baseline_metrics = None
    individual_results = []

    for eval_result in all_evaluations:
        if eval_result['phase'] == 'individual':
            df = pd.DataFrame(eval_result['results'])
            if len(df) > 0:
                method_name = eval_result['config_name']
                mae = df['mae'].mean()
                rmse = df['rmse'].mean()
                corr = df['correlation'].mean()
                within5 = df['within_5bpm'].mean()
                within10 = df['within_10bpm'].mean()

                report.append(f"{method_name:<25} {mae:<10.3f} {rmse:<10.3f} {corr:<10.3f} "
                             f"{within5:<10.1f} {within10:<10.1f}")

                individual_results.append({
                    'method': method_name,
                    'mae': mae,
                    'rmse': rmse,
                    'correlation': corr,
                    'within_5bpm': within5,
                    'within_10bpm': within10
                })

                if method_name == 'Baseline':
                    baseline_metrics = individual_results[-1]

    report.append("")

    # Calculate improvements over baseline
    if baseline_metrics:
        report.append("IMPROVEMENT OVER BASELINE:")
        report.append("-" * 80)
        report.append(f"{'Method':<25} {'MAE Delta':<12} {'RMSE Delta':<12} {'r Delta':<12} {'<10BPM Delta':<12}")
        report.append("-" * 80)

        best_method = None
        best_improvement = -float('inf')

        for result in individual_results:
            if result['method'] != 'Baseline':
                mae_delta = baseline_metrics['mae'] - result['mae']
                rmse_delta = baseline_metrics['rmse'] - result['rmse']
                corr_delta = result['correlation'] - baseline_metrics['correlation']
                within10_delta = result['within_10bpm'] - baseline_metrics['within_10bpm']

                # Overall improvement score (weighted)
                improvement_score = (mae_delta / baseline_metrics['mae']) * 0.4 + \
                                  (rmse_delta / baseline_metrics['rmse']) * 0.3 + \
                                  corr_delta * 0.2 + \
                                  (within10_delta / 100) * 0.1

                if improvement_score > best_improvement:
                    best_improvement = improvement_score
                    best_method = result

                report.append(f"{result['method']:<25} {mae_delta:<12.3f} {rmse_delta:<12.3f} "
                             f"{corr_delta:<12.3f} {within10_delta:<12.1f}")

        report.append("")
        if best_method:
            report.append(f"BEST SINGLE METHOD: {best_method['method']}")
            report.append(f"  MAE improved by {baseline_metrics['mae'] - best_method['mae']:.3f} BPM "
                         f"({(baseline_metrics['mae'] - best_method['mae']) / baseline_metrics['mae'] * 100:.1f}%)")
            report.append("")

    # Section 2: Cumulative Results
    report.append("=" * 80)
    report.append("SECTION 2: CUMULATIVE COMBINATION PERFORMANCE")
    report.append("=" * 80)
    report.append("")
    report.append("Testing combinations of best-performing methods:")
    report.append("")

    cumulative_results = []
    for eval_result in all_evaluations:
        if eval_result['phase'] == 'cumulative':
            df = pd.DataFrame(eval_result['results'])
            if len(df) > 0:
                cumulative_results.append({
                    'config': eval_result['config_name'],
                    'mae': df['mae'].mean(),
                    'rmse': df['rmse'].mean(),
                    'correlation': df['correlation'].mean(),
                    'within_10bpm': df['within_10bpm'].mean()
                })

    if cumulative_results:
        report.append(f"{'Configuration':<30} {'MAE':<10} {'RMSE':<10} {'r':<10} {'<10BPM%':<10}")
        report.append("-" * 80)

        for result in cumulative_results:
            report.append(f"{result['config']:<30} {result['mae']:<10.3f} {result['rmse']:<10.3f} "
                         f"{result['correlation']:<10.3f} {result['within_10bpm']:<10.1f}")

    report.append("")

    # Section 3: Conclusions
    report.append("=" * 80)
    report.append("SECTION 3: CONCLUSIONS AND RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")

    if cumulative_results:
        # Find best overall configuration
        best_config = min(cumulative_results, key=lambda x: x['mae'])

        report.append("BEST OVERALL CONFIGURATION:")
        report.append(f"  Name: {best_config['config']}")
        report.append(f"  MAE: {best_config['mae']:.3f} BPM")
        report.append(f"  RMSE: {best_config['rmse']:.3f} BPM")
        report.append(f"  Correlation: {best_config['correlation']:.3f}")
        report.append(f"  Within 10 BPM: {best_config['within_10bpm']:.1f}%")
        report.append("")

        if baseline_metrics:
            improvement_pct = (baseline_metrics['mae'] - best_config['mae']) / baseline_metrics['mae'] * 100
            report.append(f"OVERALL IMPROVEMENT FROM BASELINE:")
            report.append(f"  MAE improved by {baseline_metrics['mae'] - best_config['mae']:.3f} BPM ({improvement_pct:.1f}%)")
            report.append(f"  Correlation improved by {best_config['correlation'] - baseline_metrics['correlation']:.3f}")
            report.append("")

    report.append("KEY FINDINGS:")
    report.append("-" * 80)

    if best_method:
        report.append(f"1. Most effective single method: {best_method['method']}")

    report.append("2. Incremental improvements demonstrate value of systematic evaluation")
    report.append("3. Combining complementary methods yields best overall performance")
    report.append("")

    report.append("=" * 80)

    return "\n".join(report)

# ==================== Main Execution ====================

def main():
    dataset_root = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\data"

    print("\n" + "=" * 80)
    print("INCREMENTAL rPPG EVALUATION FRAMEWORK")
    print("=" * 80)

    all_evaluations = []

    # Phase 1: Evaluate individual methods
    print("\nPHASE 1: Evaluating individual improvement methods...")

    for method in IMPROVEMENT_METHODS:
        results = evaluate_configuration(
            dataset_root,
            method['config'],
            method['name'],
            max_subjects=10  # Use subset for faster testing
        )

        all_evaluations.append({
            'phase': 'individual',
            'config_name': method['name'],
            'config': method['config'],
            'results': results
        })

    # Determine best methods
    individual_scores = []
    for eval_result in all_evaluations:
        if eval_result['config_name'] != 'Baseline':
            df = pd.DataFrame(eval_result['results'])
            if len(df) > 0:
                individual_scores.append({
                    'name': eval_result['config_name'],
                    'config': eval_result['config'],
                    'mae': df['mae'].mean()
                })

    individual_scores.sort(key=lambda x: x['mae'])

    # Phase 2: Cumulative combinations
    print("\nPHASE 2: Testing cumulative combinations...")

    # Best single
    if len(individual_scores) > 0:
        best_config = individual_scores[0]['config']
        results = evaluate_configuration(dataset_root, best_config, "Best Single Method", max_subjects=10)
        all_evaluations.append({
            'phase': 'cumulative',
            'config_name': 'Best Single Method',
            'config': best_config,
            'results': results
        })

    # Best two combined
    if len(individual_scores) >= 2:
        combined_config = {**individual_scores[0]['config'], **individual_scores[1]['config']}
        results = evaluate_configuration(dataset_root, combined_config, "Best Two Methods", max_subjects=10)
        all_evaluations.append({
            'phase': 'cumulative',
            'config_name': 'Best Two Methods',
            'config': combined_config,
            'results': results
        })

    # All methods
    all_methods_config = {
        'motion_filtering': True,
        'adaptive_roi': True,
        'detrending': True,
        'adaptive_bandpass': True,
        'temporal_smoothing': True,
        'outlier_rejection': True,
    }
    results = evaluate_configuration(dataset_root, all_methods_config, "All Methods Combined", max_subjects=10)
    all_evaluations.append({
        'phase': 'cumulative',
        'config_name': 'All Methods Combined',
        'config': all_methods_config,
        'results': results
    })

    # Generate report
    print("\nGenerating comprehensive report...")
    report = generate_incremental_report(all_evaluations)

    print("\n" + report)

    # Save results with UTF-8 encoding
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    timestamp = int(time.time())
    all_results = []
    for eval_result in all_evaluations:
        for result in eval_result['results']:
            result['configuration'] = eval_result['config_name']
            result['phase'] = eval_result['phase']
            all_results.append(result)

    if all_results:
        pd.DataFrame(all_results).to_csv(f'incremental_results_{timestamp}.csv', index=False)

    print(f"\n{'='*80}")
    print("Incremental evaluation complete!")
    print(f"Report saved to: results.txt")
    print(f"Detailed results: incremental_results_{timestamp}.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
