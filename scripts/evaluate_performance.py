"""
Performance evaluation script for rPPG algorithms
Compares predicted HR with ground truth from PURE dataset
"""
import os
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from simple_rppg_ui import RPPGProcessor
import pandas as pd

class RPPGEvaluator:
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.processor = RPPGProcessor()
        
    def load_pure_ground_truth(self, session_path):
        """Load ground truth HR from PURE dataset JSON file"""
        json_file = session_path / f"{session_path.name}.json"
        if not json_file.exists():
            return None
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract HR values (usually in '/pulse' key)
        if '/pulse' in data:
            hr_values = data['/pulse']
            return np.array(hr_values)
        elif 'HR' in data:
            return np.array(data['HR'])
        else:
            # Try to find HR data in any numeric array
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    if all(isinstance(x, (int, float)) for x in value):
                        # Check if values are in reasonable HR range
                        arr = np.array(value)
                        if 40 <= np.mean(arr) <= 180:  # Reasonable HR range
                            return arr
        return None
    
    def process_video_file(self, video_path):
        """Process video file and extract HR predictions"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        print(f"Processing {video_path.name}: {duration:.1f}s, {fps:.1f} FPS")
        
        # Reset processor
        self.processor = RPPGProcessor(fps=fps)
        hr_predictions = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            hr, face_rect, filtered_signal = self.processor.process_frame(frame)
            
            # Store HR every second (when we have enough data)
            if frame_idx % int(fps) == 0 and hr > 0:
                hr_predictions.append(hr)
            
            frame_idx += 1
            
            # Show progress
            if frame_idx % (int(fps) * 10) == 0:  # Every 10 seconds
                print(f"  Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
        
        cap.release()
        return hr_predictions
    
    def calculate_metrics(self, predictions, ground_truth):
        """Calculate performance metrics"""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return {}
        
        # Align arrays (take minimum length)
        min_len = min(len(predictions), len(ground_truth))
        pred = np.array(predictions[:min_len])
        gt = np.array(ground_truth[:min_len])
        
        # Remove invalid values
        valid_mask = (pred > 0) & (gt > 0) & (pred < 200) & (gt < 200)
        pred = pred[valid_mask]
        gt = gt[valid_mask]
        
        if len(pred) == 0:
            return {}
        
        # Calculate metrics
        mae = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        mape = np.mean(np.abs((pred - gt) / gt)) * 100
        
        # Correlation
        correlation = np.corrcoef(pred, gt)[0, 1] if len(pred) > 1 else 0
        
        # Bland-Altman statistics
        mean_hr = (pred + gt) / 2
        diff_hr = pred - gt
        bias = np.mean(diff_hr)
        std_diff = np.std(diff_hr)
        limits_of_agreement = (bias - 1.96 * std_diff, bias + 1.96 * std_diff)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'correlation': correlation,
            'bias': bias,
            'std_diff': std_diff,
            'loa_lower': limits_of_agreement[0],
            'loa_upper': limits_of_agreement[1],
            'num_samples': len(pred)
        }
    
    def evaluate_dataset(self, max_sessions=None):
        """Evaluate on PURE dataset"""
        pure_path = self.dataset_root / "PURE"
        if not pure_path.exists():
            print(f"PURE dataset not found at {pure_path}")
            return {}
        
        results = []
        session_dirs = sorted([d for d in pure_path.iterdir() if d.is_dir()])
        
        if max_sessions:
            session_dirs = session_dirs[:max_sessions]
        
        print(f"Evaluating on {len(session_dirs)} PURE sessions...")
        
        for session_dir in session_dirs:
            print(f"\nProcessing session: {session_dir.name}")
            
            # Find video file
            video_files = list(session_dir.glob("*.avi"))
            if not video_files:
                print(f"  No video file found in {session_dir.name}")
                continue
            
            video_path = video_files[0]
            
            # Load ground truth
            ground_truth = self.load_pure_ground_truth(session_dir)
            if ground_truth is None:
                print(f"  No ground truth found for {session_dir.name}")
                continue
            
            # Process video
            predictions = self.process_video_file(video_path)
            
            if len(predictions) == 0:
                print(f"  No predictions generated for {session_dir.name}")
                continue
            
            # Calculate metrics
            metrics = self.calculate_metrics(predictions, ground_truth)
            if metrics:
                metrics['session'] = session_dir.name
                metrics['num_predictions'] = len(predictions)
                metrics['num_ground_truth'] = len(ground_truth)
                results.append(metrics)
                
                print(f"  Results: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, "
                      f"MAPE={metrics['mape']:.2f}%, r={metrics['correlation']:.3f}")
        
        return results
    
    def generate_report(self, results, save_path="evaluation_report.txt"):
        """Generate evaluation report"""
        if not results:
            print("No results to report")
            return
        
        # Calculate overall statistics
        metrics_df = pd.DataFrame(results)
        
        report = []
        report.append("=" * 60)
        report.append("rPPG PERFORMANCE EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Dataset: PURE")
        report.append(f"Sessions evaluated: {len(results)}")
        report.append(f"Total samples: {metrics_df['num_samples'].sum()}")
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE:")
        report.append("-" * 30)
        for metric in ['mae', 'rmse', 'mape', 'correlation']:
            values = metrics_df[metric].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                report.append(f"{metric.upper():12}: {mean_val:.3f} ± {std_val:.3f}")
        
        report.append("")
        report.append("BIAS ANALYSIS:")
        report.append("-" * 30)
        bias_values = metrics_df['bias'].dropna()
        if len(bias_values) > 0:
            mean_bias = bias_values.mean()
            report.append(f"Mean Bias:   {mean_bias:.3f} BPM")
            report.append(f"Std Diff:    {metrics_df['std_diff'].mean():.3f} BPM")
        
        # Session details
        report.append("")
        report.append("SESSION DETAILS:")
        report.append("-" * 60)
        report.append(f"{'Session':<10} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'r':<8} {'Samples':<8}")
        report.append("-" * 60)
        
        for result in results:
            report.append(f"{result['session']:<10} "
                         f"{result['mae']:<8.2f} "
                         f"{result['rmse']:<8.2f} "
                         f"{result['mape']:<8.1f} "
                         f"{result['correlation']:<8.3f} "
                         f"{result['num_samples']:<8d}")
        
        report_text = "\n".join(report)
        
        # Print report
        print("\n" + report_text)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {save_path}")
        
        return report_text
    
    def plot_results(self, results, save_path="evaluation_plots.png"):
        """Plot evaluation results"""
        if not results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('rPPG Performance Evaluation', fontsize=16)
        
        metrics_df = pd.DataFrame(results)
        
        # MAE distribution
        axes[0, 0].hist(metrics_df['mae'].dropna(), bins=10, alpha=0.7, color='blue')
        axes[0, 0].set_title('Mean Absolute Error Distribution')
        axes[0, 0].set_xlabel('MAE (BPM)')
        axes[0, 0].set_ylabel('Frequency')
        
        # RMSE vs MAE
        axes[0, 1].scatter(metrics_df['mae'], metrics_df['rmse'], alpha=0.7)
        axes[0, 1].set_xlabel('MAE (BPM)')
        axes[0, 1].set_ylabel('RMSE (BPM)')
        axes[0, 1].set_title('RMSE vs MAE')
        
        # Correlation distribution
        corr_values = metrics_df['correlation'].dropna()
        if len(corr_values) > 0:
            axes[1, 0].hist(corr_values, bins=10, alpha=0.7, color='green')
            axes[1, 0].set_title('Correlation Coefficient Distribution')
            axes[1, 0].set_xlabel('Correlation (r)')
            axes[1, 0].set_ylabel('Frequency')
        
        # Performance vs number of samples
        axes[1, 1].scatter(metrics_df['num_samples'], metrics_df['mae'], alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Number of Samples')
        axes[1, 1].set_ylabel('MAE (BPM)')
        axes[1, 1].set_title('Performance vs Sample Size')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plots saved to: {save_path}")

def main():
    # Set your dataset path
    dataset_root = r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data"
    
    evaluator = RPPGEvaluator(dataset_root)
    
    # Run evaluation on a few sessions (set to None for all)
    results = evaluator.evaluate_dataset(max_sessions=3)  # Start with 3 sessions for testing
    
    if results:
        # Generate report
        evaluator.generate_report(results)
        
        # Plot results
        evaluator.plot_results(results)
    else:
        print("No evaluation results generated")

if __name__ == "__main__":
    main()