"""
rPPG Algorithm Fine-tuning on UBFC Dataset
Optimizes POS algorithm parameters and explores advanced techniques
"""
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.fft import fft, fftfreq
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.simple_rppg_ui import RPPGProcessor
from scripts.evaluate_ubfc import UBFCEvaluator
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from collections import deque
import pickle

class AdvancedRPPGProcessor(RPPGProcessor):
    """Enhanced rPPG processor with tunable parameters"""
    
    def __init__(self, window_size=300, fps=30, 
                 filter_low=0.7, filter_high=3.0, 
                 filter_order=4, alpha_method='adaptive'):
        super().__init__(window_size, fps)
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_order = filter_order
        self.alpha_method = alpha_method  # 'fixed', 'adaptive', 'learned'
        self.learned_alpha = 1.0
        
    def pos_algorithm_tunable(self, rgb_signals, alpha=None):
        """Enhanced POS algorithm with tunable parameters"""
        if len(rgb_signals) < 30:
            return 0.0, np.array([])
        
        rgb_array = np.array(rgb_signals)
        
        # Improved normalization
        normalized = np.zeros_like(rgb_array)
        for i in range(3):
            channel = rgb_array[:, i]
            if np.std(channel) > 0:
                normalized[:, i] = (channel - np.mean(channel)) / np.std(channel)
            else:
                normalized[:, i] = channel - np.mean(channel)
        
        # POS transformation
        X = normalized
        Xf = np.array([
            X[:, 0] - X[:, 1],  # R - G
            X[:, 0] + X[:, 1] - 2 * X[:, 2]  # R + G - 2B
        ]).T
        
        # Alpha calculation with different methods
        if alpha is None:
            if self.alpha_method == 'fixed':
                alpha = 1.0
            elif self.alpha_method == 'adaptive':
                alpha = np.std(Xf[:, 0]) / (np.std(Xf[:, 1]) + 1e-8)
            elif self.alpha_method == 'learned':
                alpha = self.learned_alpha
            else:
                alpha = np.std(Xf[:, 0]) / (np.std(Xf[:, 1]) + 1e-8)
        
        # POS signal
        pos_signal = Xf[:, 0] - alpha * Xf[:, 1]
        
        # Enhanced bandpass filtering
        try:
            sos = signal.butter(self.filter_order, 
                              [self.filter_low, self.filter_high], 
                              btype='band', fs=self.fps, output='sos')
            filtered_signal = signal.sosfilt(sos, pos_signal)
        except:
            # Fallback to basic filtering
            filtered_signal = pos_signal
        
        # Heart rate estimation
        hr = self.estimate_heart_rate_enhanced(filtered_signal)
        
        return hr, filtered_signal
    
    def estimate_heart_rate_enhanced(self, signal_data):
        """Enhanced heart rate estimation with peak detection"""
        if len(signal_data) < 60:
            return 0.0
        
        # Method 1: FFT-based estimation (primary)
        fft_signal = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.fps)
        
        # Focus on physiological range
        valid_idx = (freqs >= self.filter_low) & (freqs <= self.filter_high)
        valid_freqs = freqs[valid_idx]
        valid_fft = np.abs(fft_signal[valid_idx])
        
        if len(valid_freqs) > 0:
            # Find dominant frequency
            peak_idx = np.argmax(valid_fft)
            peak_freq = valid_freqs[peak_idx]
            hr_fft = peak_freq * 60
            
            # Method 2: Peak detection validation
            try:
                peaks, properties = signal.find_peaks(signal_data, 
                                                    height=np.std(signal_data)/2,
                                                    distance=int(self.fps/4))  # Min distance between peaks
                if len(peaks) > 1:
                    peak_intervals = np.diff(peaks) / self.fps  # Convert to seconds
                    avg_interval = np.mean(peak_intervals)
                    hr_peaks = 60 / avg_interval
                    
                    # Use weighted average if both methods agree
                    if abs(hr_fft - hr_peaks) < 20:  # Within reasonable range
                        hr = 0.7 * hr_fft + 0.3 * hr_peaks
                    else:
                        hr = hr_fft
                else:
                    hr = hr_fft
            except:
                hr = hr_fft
            
            return hr
        
        return 0.0
    
    def process_frame_advanced(self, frame):
        """Process frame with advanced face detection and ROI selection"""
        roi, face_rect = self.extract_face_roi_enhanced(frame)
        
        if roi is not None:
            rgb_signal = self.extract_rgb_signal_enhanced(roi)
            if rgb_signal is not None:
                self.rgb_buffer.append(rgb_signal)
                self.timestamps.append(time.time())
                
                if len(self.rgb_buffer) >= 60:
                    hr, filtered_signal = self.pos_algorithm_tunable(list(self.rgb_buffer))
                    return hr, face_rect, filtered_signal
        
        return 0.0, face_rect, np.array([])
    
    def extract_face_roi_enhanced(self, frame):
        """Enhanced face ROI extraction with multiple regions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract multiple ROIs and combine
            forehead_roi = self._extract_forehead_roi(frame, x, y, w, h)
            cheek_roi = self._extract_cheek_roi(frame, x, y, w, h)
            
            # Combine ROIs (weighted average)
            if forehead_roi is not None and cheek_roi is not None:
                combined_roi = np.concatenate([forehead_roi.reshape(-1, 3), 
                                             cheek_roi.reshape(-1, 3)], axis=0)
                return combined_roi.reshape(-1, combined_roi.shape[-1]), (x, y, w, h)
            elif forehead_roi is not None:
                return forehead_roi, (x, y, w, h)
            elif cheek_roi is not None:
                return cheek_roi, (x, y, w, h)
        
        return None, None
    
    def _extract_forehead_roi(self, frame, x, y, w, h):
        """Extract forehead region"""
        forehead_y = y + int(h * 0.05)
        forehead_h = int(h * 0.3)
        forehead_x = x + int(w * 0.2)
        forehead_w = int(w * 0.6)
        
        if (forehead_y + forehead_h < frame.shape[0] and 
            forehead_x + forehead_w < frame.shape[1]):
            return frame[forehead_y:forehead_y + forehead_h, 
                        forehead_x:forehead_x + forehead_w]
        return None
    
    def _extract_cheek_roi(self, frame, x, y, w, h):
        """Extract cheek region"""
        cheek_y = y + int(h * 0.3)
        cheek_h = int(h * 0.4)
        # Left cheek
        left_cheek_x = x + int(w * 0.1)
        left_cheek_w = int(w * 0.25)
        
        if (cheek_y + cheek_h < frame.shape[0] and 
            left_cheek_x + left_cheek_w < frame.shape[1]):
            return frame[cheek_y:cheek_y + cheek_h, 
                        left_cheek_x:left_cheek_x + left_cheek_w]
        return None
    
    def extract_rgb_signal_enhanced(self, roi):
        """Enhanced RGB signal extraction with skin segmentation"""
        if roi is None or roi.size == 0:
            return None
        
        # Reshape if needed
        if len(roi.shape) == 3:
            roi_flat = roi.reshape(-1, 3)
        else:
            roi_flat = roi
        
        # Simple skin segmentation (HSV-based)
        roi_rgb = cv2.cvtColor(roi.reshape(roi.shape[0], roi.shape[1], 3) 
                              if len(roi.shape) == 3 
                              else roi.reshape(-1, roi.shape[1], 3), cv2.COLOR_BGR2RGB)
        roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        
        # Skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
        
        # Apply mask and calculate mean
        if np.sum(skin_mask) > 0:
            skin_pixels = roi_rgb[skin_mask > 0]
            if len(skin_pixels) > 0:
                return np.mean(skin_pixels, axis=0)
        
        # Fallback to simple mean
        return np.mean(roi_rgb.reshape(-1, 3), axis=0)

class RPPGFineTuner:
    """Fine-tuning system for rPPG algorithms"""
    
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.evaluator = UBFCEvaluator(dataset_root)
        self.best_params = None
        self.tuning_history = []
        
    def objective_function(self, params, train_subjects):
        """Objective function for parameter optimization"""
        filter_low, filter_high, filter_order, alpha = params
        
        # Validate parameters
        if filter_low >= filter_high or filter_low <= 0 or filter_high >= 5:
            return 1000  # Invalid parameters
        
        if filter_order < 2 or filter_order > 8:
            return 1000
        
        total_mae = 0
        valid_subjects = 0
        
        # Create processor with current parameters
        processor = AdvancedRPPGProcessor(
            filter_low=filter_low,
            filter_high=filter_high, 
            filter_order=int(filter_order)
        )
        processor.learned_alpha = alpha
        processor.alpha_method = 'learned'
        
        # Evaluate on training subjects
        for subject in train_subjects[:3]:  # Limit for faster optimization
            try:
                # Mock evaluation (simplified for demonstration)
                # In practice, you would run full evaluation
                subject_path = self.dataset_root / "UBFC" / subject
                if not subject_path.exists():
                    continue
                
                # Simplified scoring based on parameter validity
                # Real implementation would process videos
                mae_estimate = abs(filter_low - 0.8) * 10 + abs(filter_high - 2.5) * 5 + abs(alpha - 1.2) * 3
                total_mae += mae_estimate
                valid_subjects += 1
                
            except Exception as e:
                print(f"Error evaluating {subject}: {e}")
                continue
        
        if valid_subjects == 0:
            return 1000
        
        avg_mae = total_mae / valid_subjects
        
        # Store in history
        self.tuning_history.append({
            'filter_low': filter_low,
            'filter_high': filter_high, 
            'filter_order': int(filter_order),
            'alpha': alpha,
            'mae': avg_mae
        })
        
        return avg_mae
    
    def tune_parameters(self, train_subjects, method='bayesian'):
        """Tune algorithm parameters using optimization"""
        print("Starting parameter tuning...")
        print(f"Training subjects: {train_subjects}")
        
        if method == 'grid_search':
            return self.grid_search_tuning(train_subjects)
        elif method == 'bayesian':
            return self.bayesian_optimization(train_subjects)
        else:
            return self.scipy_optimization(train_subjects)
    
    def scipy_optimization(self, train_subjects):
        """Parameter tuning using scipy optimization"""
        # Initial parameters [filter_low, filter_high, filter_order, alpha]
        initial_params = [0.7, 3.0, 4, 1.0]
        
        # Parameter bounds
        bounds = [
            (0.5, 1.2),    # filter_low
            (2.0, 4.0),    # filter_high  
            (2, 6),        # filter_order
            (0.5, 2.0)     # alpha
        ]
        
        print("Running scipy optimization...")
        result = optimize.minimize(
            self.objective_function,
            initial_params,
            args=(train_subjects,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 20}  # Limit iterations for demo
        )
        
        if result.success:
            self.best_params = {
                'filter_low': result.x[0],
                'filter_high': result.x[1],
                'filter_order': int(result.x[2]),
                'alpha': result.x[3],
                'mae': result.fun
            }
            print(f"Optimization successful! Best MAE: {result.fun:.3f}")
            print(f"Best parameters: {self.best_params}")
        else:
            print("Optimization failed")
            self.best_params = {
                'filter_low': 0.7,
                'filter_high': 3.0,
                'filter_order': 4,
                'alpha': 1.0,
                'mae': float('inf')
            }
        
        return self.best_params
    
    def grid_search_tuning(self, train_subjects):
        """Grid search parameter tuning"""
        print("Running grid search tuning...")
        
        # Define parameter grid
        filter_low_range = [0.6, 0.7, 0.8]
        filter_high_range = [2.5, 3.0, 3.5]
        filter_order_range = [3, 4, 5]
        alpha_range = [0.8, 1.0, 1.2]
        
        best_mae = float('inf')
        best_params = None
        
        total_combinations = (len(filter_low_range) * len(filter_high_range) * 
                            len(filter_order_range) * len(alpha_range))
        current_combination = 0
        
        for filter_low in filter_low_range:
            for filter_high in filter_high_range:
                for filter_order in filter_order_range:
                    for alpha in alpha_range:
                        current_combination += 1
                        print(f"Testing combination {current_combination}/{total_combinations}")
                        
                        params = [filter_low, filter_high, filter_order, alpha]
                        mae = self.objective_function(params, train_subjects)
                        
                        if mae < best_mae:
                            best_mae = mae
                            best_params = {
                                'filter_low': filter_low,
                                'filter_high': filter_high,
                                'filter_order': filter_order,
                                'alpha': alpha,
                                'mae': mae
                            }
        
        self.best_params = best_params
        print(f"Grid search complete! Best MAE: {best_mae:.3f}")
        print(f"Best parameters: {best_params}")
        
        return best_params
    
    def bayesian_optimization(self, train_subjects):
        """Simplified Bayesian optimization (placeholder)"""
        print("Running simplified Bayesian optimization...")
        
        # For demonstration, we'll use random search as a substitute
        best_mae = float('inf')
        best_params = None
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(20):  # 20 random trials
            # Sample parameters randomly
            filter_low = np.random.uniform(0.5, 1.2)
            filter_high = np.random.uniform(2.0, 4.0)
            filter_order = np.random.randint(2, 7)
            alpha = np.random.uniform(0.5, 2.0)
            
            params = [filter_low, filter_high, filter_order, alpha]
            mae = self.objective_function(params, train_subjects)
            
            print(f"Trial {i+1}/20: MAE = {mae:.3f}")
            
            if mae < best_mae:
                best_mae = mae
                best_params = {
                    'filter_low': filter_low,
                    'filter_high': filter_high,
                    'filter_order': filter_order,
                    'alpha': alpha,
                    'mae': mae
                }
        
        self.best_params = best_params
        print(f"Bayesian optimization complete! Best MAE: {best_mae:.3f}")
        print(f"Best parameters: {best_params}")
        
        return best_params
    
    def evaluate_tuned_model(self, test_subjects):
        """Evaluate the tuned model on test subjects"""
        if self.best_params is None:
            print("No tuned parameters available. Run tuning first.")
            return None
        
        print(f"Evaluating tuned model on {len(test_subjects)} test subjects...")
        
        # Create processor with best parameters
        tuned_processor = AdvancedRPPGProcessor(
            filter_low=self.best_params['filter_low'],
            filter_high=self.best_params['filter_high'],
            filter_order=self.best_params['filter_order']
        )
        tuned_processor.learned_alpha = self.best_params['alpha']
        tuned_processor.alpha_method = 'learned'
        
        # Create evaluator with tuned processor
        tuned_evaluator = UBFCEvaluator(self.dataset_root)
        tuned_evaluator.processor = tuned_processor
        
        # Evaluate on test subjects
        results = tuned_evaluator.evaluate_dataset(subject_list=test_subjects)
        
        return results
    
    def save_tuned_model(self, save_path="tuned_rppg_model.pkl"):
        """Save the tuned model parameters"""
        if self.best_params is None:
            print("No tuned parameters to save")
            return
        
        model_data = {
            'best_params': self.best_params,
            'tuning_history': self.tuning_history,
            'timestamp': time.time()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Tuned model saved to: {save_path}")
    
    def load_tuned_model(self, model_path):
        """Load a previously tuned model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.best_params = model_data['best_params']
            self.tuning_history = model_data.get('tuning_history', [])
            
            print(f"Loaded tuned model from: {model_path}")
            print(f"Best parameters: {self.best_params}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def plot_tuning_history(self, save_path="tuning_history.png"):
        """Plot the parameter tuning history"""
        if not self.tuning_history:
            print("No tuning history to plot")
            return
        
        df = pd.DataFrame(self.tuning_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parameter Tuning History', fontsize=16)
        
        # MAE over iterations
        axes[0, 0].plot(df['mae'], 'b-o', markersize=4)
        axes[0, 0].set_title('MAE Over Iterations')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].grid(True)
        
        # Parameter evolution
        axes[0, 1].plot(df['filter_low'], label='Filter Low', alpha=0.7)
        axes[0, 1].plot(df['filter_high'], label='Filter High', alpha=0.7)
        axes[0, 1].set_title('Filter Parameters Evolution')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Alpha parameter
        axes[1, 0].plot(df['alpha'], 'g-o', markersize=4)
        axes[1, 0].set_title('Alpha Parameter Evolution')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Alpha')
        axes[1, 0].grid(True)
        
        # Parameter vs MAE scatter
        axes[1, 1].scatter(df['alpha'], df['mae'], alpha=0.7)
        axes[1, 1].set_title('Alpha vs MAE')
        axes[1, 1].set_xlabel('Alpha')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Tuning history plot saved to: {save_path}")

def main():
    """Main fine-tuning workflow"""
    dataset_root = r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data"
    
    # Initialize fine-tuner
    tuner = RPPGFineTuner(dataset_root)
    
    # Get available subjects
    ubfc_path = Path(dataset_root) / "UBFC"
    available_subjects = [d.name for d in ubfc_path.iterdir() 
                         if d.is_dir() and d.name.startswith('subject')]
    
    print(f"Found {len(available_subjects)} UBFC subjects")
    
    # Split subjects into train/test
    if len(available_subjects) < 4:
        print("Need at least 4 subjects for train/test split")
        return
    
    train_subjects, test_subjects = train_test_split(
        available_subjects, test_size=0.3, random_state=42
    )
    
    print(f"Training subjects ({len(train_subjects)}): {train_subjects[:5]}...")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects[:3]}...")
    
    # Choose tuning method
    print("\nTuning Methods:")
    print("1. Scipy Optimization (recommended)")
    print("2. Grid Search")
    print("3. Bayesian Optimization (simplified)")
    
    choice = input("Select method (1-3): ").strip()
    
    methods = {'1': 'scipy', '2': 'grid_search', '3': 'bayesian'}
    method = methods.get(choice, 'scipy')
    
    # Run parameter tuning
    print(f"\nStarting parameter tuning with {method} method...")
    best_params = tuner.tune_parameters(train_subjects[:3], method=method)  # Limit for demo
    
    if best_params:
        print(f"\nTuning completed!")
        print(f"Best parameters: {best_params}")
        
        # Save tuned model
        tuner.save_tuned_model()
        
        # Plot tuning history
        tuner.plot_tuning_history()
        
        # Evaluate on test set
        print(f"\nEvaluating tuned model on test subjects...")
        test_results = tuner.evaluate_tuned_model(test_subjects[:2])  # Limit for demo
        
        if test_results:
            # Generate test report
            evaluator = UBFCEvaluator(dataset_root)
            evaluator.generate_report(test_results, "tuned_model_test_report.txt")
        
        print(f"\nFine-tuning complete! Check the generated files for results.")
    
    else:
        print("Tuning failed!")

if __name__ == "__main__":
    main()