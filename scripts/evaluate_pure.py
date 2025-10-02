"""
PURE Dataset Evaluation
Evaluates rPPG algorithms on PURE dataset with image sequences and JSON ground truth
"""
import os
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.simple_rppg_ui import RPPGProcessor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import time
import json

class PUREEvaluator:
    def __init__(self, dataset_root, fps=30):
        self.dataset_root = Path(dataset_root)
        self.fps = fps
        self.processor = RPPGProcessor(fps=fps)

    def load_pure_ground_truth(self, subject_path):
        """Load ground truth HR from PURE subject JSON file"""
        # Find JSON file
        json_files = list(subject_path.glob("*.json"))
        if not json_files:
            return None

        json_file = json_files[0]

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract heart rate values from JSON
            hr_values = []
            timestamps = []

            if '/FullPackage' in data:
                for entry in data['/FullPackage']:
                    if 'Value' in entry and 'pulseRate' in entry['Value']:
                        hr = entry['Value']['pulseRate']
                        if hr > 0:  # Only valid readings
                            hr_values.append(hr)
                            if 'Timestamp' in entry:
                                timestamps.append(entry['Timestamp'])

            if len(hr_values) == 0:
                return None

            return np.array(hr_values), np.array(timestamps)

        except Exception as e:
            print(f"Error loading ground truth from {json_file}: {e}")
            return None

    def process_pure_images(self, subject_path, subject_name):
        """Process PURE image sequence and extract HR predictions"""
        # Get subdirectory with images (same name as subject)
        image_dir = subject_path / subject_name
        if not image_dir.exists():
            print(f"Image directory not found: {image_dir}")
            return None, None

        # Get all PNG images
        image_files = sorted(image_dir.glob("*.png"))
        if not image_files:
            print(f"No images found in {image_dir}")
            return None, None

        print(f"Processing {subject_name}: {len(image_files)} images at {self.fps} FPS")

        # Reset processor for this subject
        self.processor = RPPGProcessor(fps=self.fps)
        hr_predictions = []
        timestamps = []

        for idx, image_path in enumerate(image_files):
            # Read image
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Process frame
            hr, face_rect, filtered_signal = self.processor.process_frame(frame)

            # Store prediction and timestamp
            current_time = idx / self.fps
            hr_predictions.append(hr if hr > 0 else np.nan)
            timestamps.append(current_time)

            # Progress update
            if (idx + 1) % (self.fps * 30) == 0:  # Every 30 seconds
                print(f"  Processed {idx+1}/{len(image_files)} images ({(idx+1)/len(image_files)*100:.1f}%)")

        return np.array(hr_predictions), np.array(timestamps)

    def align_predictions_with_ground_truth(self, predictions, pred_timestamps, ground_truth, gt_timestamps=None):
        """Align HR predictions with ground truth data"""
        if gt_timestamps is None:
            # If no GT timestamps, assume uniform sampling at same rate
            gt_fps = len(ground_truth) / pred_timestamps[-1] if len(pred_timestamps) > 0 else self.fps
            gt_timestamps = np.arange(len(ground_truth)) / gt_fps
        else:
            # Normalize timestamps to seconds (PURE uses nanoseconds)
            if gt_timestamps[0] > 1e12:  # Likely nanoseconds
                gt_timestamps = (gt_timestamps - gt_timestamps[0]) / 1e9

        # Find overlapping time range
        start_time = max(pred_timestamps[0], gt_timestamps[0])
        end_time = min(pred_timestamps[-1], gt_timestamps[-1])

        if start_time >= end_time:
            print("No overlapping time range found")
            return np.array([]), np.array([])

        # Create common time grid (1 second intervals)
        common_time = np.arange(start_time, end_time, 1.0)

        # Interpolate both predictions and ground truth to common time grid
        pred_interp = np.interp(common_time, pred_timestamps, predictions)
        gt_interp = np.interp(common_time, gt_timestamps, ground_truth)

        # Remove NaN values
        valid_mask = ~(np.isnan(pred_interp) | np.isnan(gt_interp))

        return pred_interp[valid_mask], gt_interp[valid_mask]

    def calculate_metrics(self, predictions, ground_truth):
        """Calculate comprehensive performance metrics"""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return {}

        # Basic metrics
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mean_squared_error(ground_truth, predictions))
        mape = np.mean(np.abs((predictions - ground_truth) / ground_truth)) * 100

        # Correlation
        if len(predictions) > 1 and np.std(predictions) > 0 and np.std(ground_truth) > 0:
            correlation, p_value = pearsonr(predictions, ground_truth)
        else:
            correlation, p_value = 0, 1

        # Bland-Altman statistics
        mean_hr = (predictions + ground_truth) / 2
        diff_hr = predictions - ground_truth
        bias = np.mean(diff_hr)
        std_diff = np.std(diff_hr)
        limits_of_agreement = (bias - 1.96 * std_diff, bias + 1.96 * std_diff)

        # Percentage of predictions within acceptable ranges
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
            'loa_lower': limits_of_agreement[0],
            'loa_upper': limits_of_agreement[1],
            'within_5bpm': within_5bpm,
            'within_10bpm': within_10bpm,
            'num_samples': len(predictions)
        }

    def evaluate_subject(self, subject_name):
        """Evaluate single PURE subject"""
        subject_path = self.dataset_root / "PURE" / subject_name

        if not subject_path.exists():
            print(f"Subject {subject_name} not found")
            return None

        # Load ground truth
        gt_result = self.load_pure_ground_truth(subject_path)
        if gt_result is None:
            print(f"No ground truth found for {subject_name}")
            return None

        ground_truth, gt_timestamps = gt_result

        # Process images
        print(f"\nEvaluating {subject_name}...")
        start_time = time.time()

        pred_result = self.process_pure_images(subject_path, subject_name)
        if pred_result is None or pred_result[0] is None:
            print(f"Failed to process images for {subject_name}")
            return None

        predictions, pred_timestamps = pred_result

        if len(predictions) == 0:
            print(f"No predictions generated for {subject_name}")
            return None

        # Align with ground truth
        aligned_pred, aligned_gt = self.align_predictions_with_ground_truth(
            predictions, pred_timestamps, ground_truth, gt_timestamps
        )

        if len(aligned_pred) == 0:
            print(f"No aligned data for {subject_name}")
            return None

        # Calculate metrics
        metrics = self.calculate_metrics(aligned_pred, aligned_gt)

        if metrics:
            metrics['subject'] = subject_name
            metrics['processing_time'] = time.time() - start_time
            metrics['video_duration'] = pred_timestamps[-1] if len(pred_timestamps) > 0 else 0

            print(f"Results: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, "
                  f"MAPE={metrics['mape']:.2f}%, r={metrics['correlation']:.3f}")
            print(f"Within 5 BPM: {metrics['within_5bpm']:.1f}%, "
                  f"Within 10 BPM: {metrics['within_10bpm']:.1f}%")

        return {
            'metrics': metrics,
            'predictions': aligned_pred,
            'ground_truth': aligned_gt,
            'timestamps': pred_timestamps
        }

    def evaluate_dataset(self, max_subjects=None, subject_list=None):
        """Evaluate on PURE dataset"""
        pure_path = self.dataset_root / "PURE"
        if not pure_path.exists():
            print(f"PURE dataset not found at {pure_path}")
            return {}

        # Get subject directories (format: XX-XX)
        if subject_list:
            subject_dirs = [pure_path / s for s in subject_list if (pure_path / s).exists()]
        else:
            subject_dirs = sorted([d for d in pure_path.iterdir()
                                 if d.is_dir() and '-' in d.name and len(d.name) == 5])

        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]

        print(f"Evaluating on {len(subject_dirs)} PURE subjects...")

        results = []
        successful_subjects = []

        for subject_dir in subject_dirs:
            subject_name = subject_dir.name
            result = self.evaluate_subject(subject_name)

            if result and result['metrics']:
                results.append(result['metrics'])
                successful_subjects.append(subject_name)
            else:
                print(f"Failed to evaluate {subject_name}")

        print(f"\nSuccessfully evaluated {len(results)}/{len(subject_dirs)} subjects")
        return {
            'results': results,
            'successful_subjects': successful_subjects
        }

    def generate_report(self, evaluation_results, save_path="pure_evaluation_report.txt"):
        """Generate comprehensive evaluation report"""
        results = evaluation_results.get('results', [])

        if not results:
            print("No results to report")
            return

        # Convert to DataFrame for analysis
        metrics_df = pd.DataFrame(results)

        report = []
        report.append("=" * 70)
        report.append("PURE DATASET EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"Dataset: PURE")
        report.append(f"Subjects evaluated: {len(results)}")
        report.append(f"Total samples: {metrics_df['num_samples'].sum()}")
        report.append(f"Algorithm: POS (Plane-Orthogonal-to-Skin)")
        report.append("")

        # Overall performance metrics
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 40)

        metrics_to_report = ['mae', 'rmse', 'mape', 'correlation', 'within_5bpm', 'within_10bpm']
        for metric in metrics_to_report:
            values = metrics_df[metric].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()

                if metric in ['within_5bpm', 'within_10bpm']:
                    report.append(f"{metric.replace('_', ' ').title():<15}: {mean_val:.1f}% ± {std_val:.1f}%")
                elif metric == 'mape':
                    report.append(f"{metric.upper():<15}: {mean_val:.2f}% ± {std_val:.2f}%")
                else:
                    report.append(f"{metric.upper():<15}: {mean_val:.3f} ± {std_val:.3f}")

        # Performance benchmarking
        report.append("")
        report.append("PERFORMANCE BENCHMARK:")
        report.append("-" * 40)
        mae_mean = metrics_df['mae'].mean()
        corr_mean = metrics_df['correlation'].mean()
        within_10_mean = metrics_df['within_10bpm'].mean()

        if mae_mean < 5 and corr_mean > 0.8 and within_10_mean > 80:
            report.append("[EXCELLENT] Clinical-grade performance")
        elif mae_mean < 8 and corr_mean > 0.7 and within_10_mean > 70:
            report.append("[GOOD] Research-grade performance")
        elif mae_mean < 12 and corr_mean > 0.5 and within_10_mean > 60:
            report.append("[ACCEPTABLE] Requires improvement")
        else:
            report.append("[POOR] Significant improvements needed")

        # Subject-by-subject results
        report.append("")
        report.append("SUBJECT-BY-SUBJECT RESULTS:")
        report.append("-" * 70)
        report.append(f"{'Subject':<12} {'MAE':<8} {'RMSE':<8} {'MAPE%':<8} {'r':<8} {'<5BPM%':<8} {'<10BPM%':<8}")
        report.append("-" * 70)

        for result in results:
            report.append(f"{result['subject']:<12} "
                         f"{result['mae']:<8.2f} "
                         f"{result['rmse']:<8.2f} "
                         f"{result['mape']:<8.1f} "
                         f"{result['correlation']:<8.3f} "
                         f"{result['within_5bpm']:<8.1f} "
                         f"{result['within_10bpm']:<8.1f}")

        # Statistical analysis
        report.append("")
        report.append("STATISTICAL ANALYSIS:")
        report.append("-" * 40)
        report.append(f"Best MAE: {metrics_df['mae'].min():.2f} BPM ({metrics_df.loc[metrics_df['mae'].idxmin(), 'subject']})")
        report.append(f"Worst MAE: {metrics_df['mae'].max():.2f} BPM ({metrics_df.loc[metrics_df['mae'].idxmax(), 'subject']})")
        report.append(f"Best Correlation: {metrics_df['correlation'].max():.3f} ({metrics_df.loc[metrics_df['correlation'].idxmax(), 'subject']})")
        report.append(f"Worst Correlation: {metrics_df['correlation'].min():.3f} ({metrics_df.loc[metrics_df['correlation'].idxmin(), 'subject']})")

        # Processing performance
        total_time = metrics_df['processing_time'].sum()
        total_duration = metrics_df['video_duration'].sum()
        report.append(f"Total video duration: {total_duration:.1f} seconds")
        report.append(f"Total processing time: {total_time:.1f} seconds")
        if total_time > 0:
            report.append(f"Processing speed: {total_duration/total_time:.1f}x realtime")

        report_text = "\n".join(report)

        # Print and save report
        print("\n" + report_text)

        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {save_path}")

        return report_text

    def plot_evaluation_results(self, evaluation_results, save_path="pure_evaluation_plots.png"):
        """Create comprehensive evaluation plots"""
        results = evaluation_results.get('results', [])

        if not results:
            print("No results to plot")
            return

        metrics_df = pd.DataFrame(results)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PURE Dataset Evaluation Results', fontsize=16)

        # MAE distribution
        axes[0, 0].hist(metrics_df['mae'], bins=15, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Mean Absolute Error Distribution')
        axes[0, 0].set_xlabel('MAE (BPM)')
        axes[0, 0].set_ylabel('Number of Subjects')
        axes[0, 0].axvline(metrics_df['mae'].mean(), color='red', linestyle='--',
                          label=f'Mean: {metrics_df["mae"].mean():.2f}')
        axes[0, 0].legend()

        # Correlation distribution
        axes[0, 1].hist(metrics_df['correlation'], bins=15, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Correlation Distribution')
        axes[0, 1].set_xlabel('Pearson Correlation (r)')
        axes[0, 1].set_ylabel('Number of Subjects')
        axes[0, 1].axvline(metrics_df['correlation'].mean(), color='red', linestyle='--',
                          label=f'Mean: {metrics_df["correlation"].mean():.3f}')
        axes[0, 1].legend()

        # Accuracy within thresholds
        within_5 = metrics_df['within_5bpm'].mean()
        within_10 = metrics_df['within_10bpm'].mean()
        within_15 = 100  # Assume all are within 15 BPM for comparison

        axes[0, 2].bar(['Within 5 BPM', 'Within 10 BPM', 'Within 15 BPM'],
                      [within_5, within_10, within_15],
                      color=['red', 'orange', 'green'], alpha=0.7)
        axes[0, 2].set_title('Accuracy Within Thresholds')
        axes[0, 2].set_ylabel('Percentage of Predictions (%)')
        axes[0, 2].set_ylim(0, 100)

        # MAE vs Correlation scatter
        axes[1, 0].scatter(metrics_df['mae'], metrics_df['correlation'], alpha=0.7, s=50)
        axes[1, 0].set_xlabel('MAE (BPM)')
        axes[1, 0].set_ylabel('Correlation (r)')
        axes[1, 0].set_title('MAE vs Correlation')

        # Subject performance ranking
        sorted_subjects = metrics_df.sort_values('mae')
        y_pos = np.arange(len(sorted_subjects))
        axes[1, 1].barh(y_pos, sorted_subjects['mae'], alpha=0.7)
        axes[1, 1].set_xlabel('MAE (BPM)')
        axes[1, 1].set_title('Subject Performance (MAE)')
        axes[1, 1].set_yticks(y_pos[::max(1, len(y_pos)//10)])  # Show max 10 labels
        axes[1, 1].set_yticklabels(sorted_subjects['subject'].iloc[::max(1, len(y_pos)//10)])

        # MAPE distribution
        axes[1, 2].hist(metrics_df['mape'], bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 2].set_title('Mean Absolute Percentage Error')
        axes[1, 2].set_xlabel('MAPE (%)')
        axes[1, 2].set_ylabel('Number of Subjects')
        axes[1, 2].axvline(metrics_df['mape'].mean(), color='red', linestyle='--',
                          label=f'Mean: {metrics_df["mape"].mean():.1f}%')
        axes[1, 2].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {save_path}")

def main():
    """Main evaluation function"""
    # Dataset configuration
    dataset_root = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\data"

    # Create evaluator
    evaluator = PUREEvaluator(dataset_root)

    # Choose evaluation scope
    print("PURE Dataset Evaluation Options:")
    print("1. Quick test (3 subjects)")
    print("2. Medium test (10 subjects)")
    print("3. Full evaluation (all subjects)")
    print("4. Custom subject list")

    choice = input("Select option (1-4): ").strip()

    if choice == "1":
        # Quick test with first 3 subjects
        results = evaluator.evaluate_dataset(max_subjects=3)
    elif choice == "2":
        # Medium test with 10 subjects
        results = evaluator.evaluate_dataset(max_subjects=10)
    elif choice == "3":
        # Full evaluation
        results = evaluator.evaluate_dataset()
    elif choice == "4":
        # Custom subject list
        subjects_input = input("Enter subject names (e.g., 01-01,01-02,02-01): ")
        subject_list = [s.strip() for s in subjects_input.split(',')]
        results = evaluator.evaluate_dataset(subject_list=subject_list)
    else:
        # Default to quick test
        print("Invalid choice, running quick test...")
        results = evaluator.evaluate_dataset(max_subjects=3)

    if results and results.get('results'):
        # Generate report
        evaluator.generate_report(results)

        # Create plots
        try:
            evaluator.plot_evaluation_results(results)
        except Exception as e:
            print(f"Could not generate plots: {e}")

        # Save detailed results
        timestamp = int(time.time())
        results_file = f"pure_results_{timestamp}.csv"
        pd.DataFrame(results['results']).to_csv(results_file, index=False)
        print(f"Detailed results saved to: {results_file}")

    else:
        print("No evaluation results generated")

if __name__ == "__main__":
    main()
