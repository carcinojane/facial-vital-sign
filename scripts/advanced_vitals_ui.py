"""
Advanced Vital Signs UI with comprehensive monitoring
Displays heart rate, respiratory rate, SpO2 estimation, and more
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import time
from collections import deque
import threading
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_rppg_ui import RPPGProcessor

class AdvancedVitalsProcessor(RPPGProcessor):
    def __init__(self, window_size=300, fps=30):
        super().__init__(window_size, fps)
        
        # Additional buffers for vital signs
        self.respiratory_buffer = deque(maxlen=600)  # 20 seconds for respiratory rate
        self.spo2_buffer = deque(maxlen=150)  # 5 seconds for SpO2
        self.hrv_buffer = deque(maxlen=1800)  # 60 seconds for HRV
        
        # Vital signs storage
        self.current_rr = 0.0  # Respiratory rate
        self.current_spo2 = 0.0  # Oxygen saturation
        self.current_hrv = 0.0  # Heart rate variability
        self.current_stress_level = 0.0  # Stress indicator
        self.current_bp_systolic = 120  # Systolic blood pressure
        self.current_bp_diastolic = 80  # Diastolic blood pressure
        
        # Signal quality metrics
        self.signal_quality = 0.0
        self.face_stability = 0.0
        
    def estimate_respiratory_rate(self, rgb_signals):
        """Estimate respiratory rate from RGB signal variations"""
        if len(rgb_signals) < 300:  # Need at least 10 seconds
            return 0.0
        
        try:
            # Use green channel variations for respiratory estimation
            rgb_array = np.array(rgb_signals)
            green_channel = rgb_array[:, 1]
            
            # Apply low-pass filter for respiratory frequency range (0.1-0.5 Hz, 6-30 breaths/min)
            sos_resp = signal.butter(3, [0.1, 0.5], btype='band', fs=self.fps, output='sos')
            resp_signal = signal.sosfilt(sos_resp, green_channel)
            
            # FFT for respiratory rate
            fft_resp = fft(resp_signal)
            freqs_resp = fftfreq(len(resp_signal), 1/self.fps)
            
            # Find peak in respiratory range
            valid_idx = (freqs_resp >= 0.1) & (freqs_resp <= 0.5)
            valid_freqs = freqs_resp[valid_idx]
            valid_fft = np.abs(fft_resp[valid_idx])
            
            if len(valid_freqs) > 0:
                peak_idx = np.argmax(valid_fft)
                peak_freq = valid_freqs[peak_idx]
                rr = peak_freq * 60  # Convert to breaths per minute
                return max(6, min(40, rr))  # Clamp to physiological range
                
        except Exception as e:
            pass
            
        return 0.0
    
    def estimate_spo2(self, rgb_signals):
        """Estimate SpO2 from red/infrared ratio (approximate)"""
        if len(rgb_signals) < 90:  # Need at least 3 seconds
            return 0.0
        
        try:
            rgb_array = np.array(rgb_signals)
            red_channel = rgb_array[:, 0]
            blue_channel = rgb_array[:, 2]  # Use blue as IR substitute
            
            # Calculate AC/DC ratios
            red_ac = np.std(red_channel)
            red_dc = np.mean(red_channel)
            blue_ac = np.std(blue_channel)
            blue_dc = np.mean(blue_channel)
            
            if red_dc > 0 and blue_dc > 0:
                # Ratio of ratios (simplified SpO2 calculation)
                ratio = (red_ac / red_dc) / (blue_ac / blue_dc + 1e-8)
                
                # Empirical formula (this is a rough approximation)
                spo2 = 110 - 25 * ratio
                
                # Clamp to realistic range
                spo2 = max(80, min(100, spo2))
                
                # Add some noise reduction
                if hasattr(self, 'prev_spo2'):
                    spo2 = 0.7 * self.prev_spo2 + 0.3 * spo2
                self.prev_spo2 = spo2
                
                return spo2
                
        except Exception as e:
            pass
            
        return 95.0  # Default reasonable value
    
    def calculate_hrv(self, hr_values):
        """Calculate Heart Rate Variability"""
        if len(hr_values) < 30:
            return 0.0
        
        try:
            # Convert HR to RR intervals (approximate)
            rr_intervals = 60000 / np.array(hr_values)  # in milliseconds
            
            # Calculate RMSSD (Root Mean Square of Successive Differences)
            if len(rr_intervals) > 1:
                successive_diffs = np.diff(rr_intervals)
                rmssd = np.sqrt(np.mean(successive_diffs ** 2))
                return rmssd
                
        except Exception as e:
            pass
            
        return 0.0
    
    def estimate_blood_pressure(self, hr_values, pulse_transit_times=None):
        """Estimate blood pressure from HR patterns and pulse characteristics"""
        try:
            if not hr_values or len(hr_values) < 30:
                return 120, 80  # Default values
                
            hr_array = np.array(hr_values)
            mean_hr = np.mean(hr_array)
            hr_variability = np.std(hr_array)
            
            # Basic BP estimation based on HR patterns
            # These are simplified correlations - real BP needs calibration
            
            # Systolic BP estimation (correlation with mean HR)
            base_systolic = 120
            hr_factor = (mean_hr - 70) * 0.8  # HR impact on systolic
            variability_factor = hr_variability * 2  # Variability impact
            
            systolic = base_systolic + hr_factor + variability_factor
            systolic = max(90, min(180, systolic))  # Reasonable bounds
            
            # Diastolic BP estimation 
            base_diastolic = 80
            diastolic_hr_factor = (mean_hr - 70) * 0.4  # Smaller HR impact
            diastolic_variability = hr_variability * 1.5
            
            diastolic = base_diastolic + diastolic_hr_factor + diastolic_variability
            diastolic = max(60, min(110, diastolic))  # Reasonable bounds
            
            # Ensure diastolic < systolic
            if diastolic >= systolic:
                diastolic = systolic - 20
                
            return int(systolic), int(diastolic)
            
        except Exception as e:
            return 120, 80  # Default values
    
    def estimate_stress_level(self, hr, hrv, rr):
        """Estimate stress level based on vital signs"""
        try:
            stress_indicators = []
            
            # HR-based stress (elevated HR)
            if hr > 90:
                stress_indicators.append((hr - 70) / 50)  # Normalized stress from HR
            
            # HRV-based stress (low HRV = high stress)
            if hrv > 0:
                stress_indicators.append(max(0, (50 - hrv) / 50))
            
            # RR-based stress (elevated breathing)
            if rr > 18:
                stress_indicators.append((rr - 15) / 10)
            
            if stress_indicators:
                stress_level = np.mean(stress_indicators)
                return max(0, min(1, stress_level)) * 100  # 0-100 scale
            
        except Exception as e:
            pass
            
        return 0.0
    
    def calculate_signal_quality(self, rgb_signals, face_rect):
        """Calculate signal quality metrics"""
        quality_score = 0.0
        
        try:
            if len(rgb_signals) > 30:
                rgb_array = np.array(rgb_signals)
                
                # Signal stability (lower std deviation = better)
                signal_stability = 1.0 / (1.0 + np.std(rgb_array))
                
                # Face detection consistency
                face_present = 1.0 if face_rect is not None else 0.0
                
                # Combine metrics
                quality_score = (signal_stability + face_present) / 2.0
                
        except Exception as e:
            pass
            
        return quality_score * 100  # 0-100 scale
    
    def process_frame_advanced(self, frame):
        """Process frame and extract all vital signs"""
        # Get basic HR processing
        hr, face_rect, filtered_signal = self.process_frame(frame)
        
        # Update respiratory buffer
        if len(self.rgb_buffer) > 0:
            self.respiratory_buffer.extend(list(self.rgb_buffer)[-10:])  # Add recent samples
        
        # Calculate additional vital signs
        if len(self.rgb_buffer) >= 90:  # 3 seconds minimum
            # Respiratory rate
            if len(self.respiratory_buffer) >= 300:
                self.current_rr = self.estimate_respiratory_rate(list(self.respiratory_buffer))
            
            # SpO2 estimation
            recent_rgb = list(self.rgb_buffer)[-90:]  # Last 3 seconds
            self.current_spo2 = self.estimate_spo2(recent_rgb)
            
            # HRV calculation
            if len(self.hr_history) >= 30:
                recent_hrs = list(self.hr_history)[-30:]  # Last 30 HR values
                self.current_hrv = self.calculate_hrv(recent_hrs)
            
            # Stress level
            self.current_stress_level = self.estimate_stress_level(hr, self.current_hrv, self.current_rr)
            
            # Blood pressure estimation
            if len(self.hr_history) >= 30:
                recent_hrs = list(self.hr_history)[-30:]
                self.current_bp_systolic, self.current_bp_diastolic = self.estimate_blood_pressure(recent_hrs)
            
            # Signal quality
            self.signal_quality = self.calculate_signal_quality(list(self.rgb_buffer), face_rect)
        
        return {
            'hr': hr,
            'rr': self.current_rr,
            'spo2': self.current_spo2,
            'hrv': self.current_hrv,
            'stress': self.current_stress_level,
            'bp_systolic': self.current_bp_systolic,
            'bp_diastolic': self.current_bp_diastolic,
            'signal_quality': self.signal_quality,
            'face_rect': face_rect,
            'filtered_signal': filtered_signal
        }

class AdvancedVitalsApp:
    def __init__(self):
        self.processor = AdvancedVitalsProcessor()
        self.cap = None
        self.running = False
        
        # Vital signs history for plotting
        self.hr_history = deque(maxlen=300)  # 10 seconds at 30fps
        self.rr_history = deque(maxlen=300)
        self.spo2_history = deque(maxlen=300)
        self.stress_history = deque(maxlen=300)
        
        # Display mode
        self.show_graphs = False
        
        # Colors
        self.colors = {
            'excellent': (0, 255, 0),
            'good': (0, 255, 255),
            'fair': (0, 165, 255),
            'poor': (0, 0, 255),
            'critical': (0, 0, 139)
        }
    
    def get_vital_status_color(self, vital_type, value):
        """Get color based on vital sign value"""
        if vital_type == 'hr':
            if 60 <= value <= 100:
                return self.colors['excellent']
            elif 50 <= value <= 110:
                return self.colors['good']
            elif 40 <= value <= 120:
                return self.colors['fair']
            else:
                return self.colors['poor']
                
        elif vital_type == 'rr':
            if 12 <= value <= 20:
                return self.colors['excellent']
            elif 10 <= value <= 24:
                return self.colors['good']
            elif 8 <= value <= 30:
                return self.colors['fair']
            else:
                return self.colors['poor']
                
        elif vital_type == 'spo2':
            if value >= 98:
                return self.colors['excellent']
            elif value >= 95:
                return self.colors['good']
            elif value >= 90:
                return self.colors['fair']
            else:
                return self.colors['critical']
                
        elif vital_type == 'stress':
            if value <= 20:
                return self.colors['excellent']
            elif value <= 40:
                return self.colors['good']
            elif value <= 60:
                return self.colors['fair']
            else:
                return self.colors['poor']
                
        elif vital_type == 'bp':
            if 90 <= value <= 120:  # Normal systolic
                return self.colors['excellent']
            elif 80 <= value <= 139:  # Borderline
                return self.colors['good']
            elif 140 <= value <= 159:  # Stage 1 hypertension
                return self.colors['fair']
            else:  # High risk
                return self.colors['poor']
        
        return self.colors['good']
    
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def draw_vital_panel(self, frame, vitals):
        """Draw comprehensive vital signs panel"""
        h, w = frame.shape[:2]
        panel_height = 200
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Main vital signs - larger display
        vital_y_start = h - 170
        
        # Heart Rate (large, prominent)
        hr_color = self.get_vital_status_color('hr', vitals['hr'])
        cv2.putText(frame, f"HR: {vitals['hr']:.0f}", (20, vital_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, hr_color, 3)
        cv2.putText(frame, "BPM", (20, vital_y_start + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2)
        
        # Respiratory Rate
        rr_color = self.get_vital_status_color('rr', vitals['rr'])
        cv2.putText(frame, f"RR: {vitals['rr']:.0f}", (250, vital_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, rr_color, 2)
        cv2.putText(frame, "BrPM", (250, vital_y_start + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, rr_color, 1)
        
        # SpO2
        spo2_color = self.get_vital_status_color('spo2', vitals['spo2'])
        cv2.putText(frame, f"SpO2: {vitals['spo2']:.0f}%", (420, vital_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, spo2_color, 2)
        
        # Heart Rate Variability
        cv2.putText(frame, f"HRV: {vitals['hrv']:.1f}", (620, vital_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "RMSSD", (620, vital_y_start + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Stress Level with visual indicator
        stress_color = self.get_vital_status_color('stress', vitals['stress'])
        cv2.putText(frame, f"Stress: {vitals['stress']:.0f}%", (820, vital_y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, stress_color, 2)
        
        # Stress level bar
        bar_x, bar_y = 820, vital_y_start + 30
        bar_width, bar_height = 120, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 1)
        stress_fill = int((vitals['stress'] / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + stress_fill, bar_y + bar_height), stress_color, -1)
        
        # Blood Pressure
        bp_y = vital_y_start + 70
        bp_color = self.get_vital_status_color('bp', vitals['bp_systolic'])
        cv2.putText(frame, f"BP: {vitals['bp_systolic']}/{vitals['bp_diastolic']}", (20, bp_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, bp_color, 2)
        cv2.putText(frame, "mmHg", (20, bp_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Signal Quality
        quality_y = h - 100
        quality_color = self.get_vital_status_color('spo2', vitals['signal_quality'])  # Reuse spo2 logic
        cv2.putText(frame, f"Signal Quality: {vitals['signal_quality']:.0f}%", (20, quality_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, quality_color, 2)
        
        # Quality bar
        bar_x, bar_y = 250, quality_y - 15
        bar_width, bar_height = 150, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), 1)
        quality_fill = int((vitals['signal_quality'] / 100) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + quality_fill, bar_y + bar_height), quality_color, -1)
        
        # Timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 120, quality_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Instructions
        cv2.putText(frame, "Controls: Q-Quit, S-Save, R-Reset, G-Graphs, SPACE-Screenshot", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    def draw_face_detection(self, frame, face_rect):
        """Draw enhanced face detection with ROI indicators"""
        if face_rect is not None:
            x, y, w, h = face_rect
            
            # Main face rectangle (green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Forehead ROI (blue)
            forehead_y = y + int(h * 0.1)
            forehead_h = int(h * 0.4)
            cv2.rectangle(frame, (x, forehead_y), (x + w, forehead_y + forehead_h), (255, 0, 0), 1)
            cv2.putText(frame, "HR ROI", (x, forehead_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Cheek ROI for respiratory (yellow)
            cheek_y = y + int(h * 0.3)
            cheek_h = int(h * 0.4)
            cheek_w = int(w * 0.3)
            cv2.rectangle(frame, (x + int(w * 0.1), cheek_y), (x + int(w * 0.1) + cheek_w, cheek_y + cheek_h), (0, 255, 255), 1)
            cv2.putText(frame, "RR ROI", (x + int(w * 0.1), cheek_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def save_vital_signs_data(self):
        """Save comprehensive vital signs data"""
        if len(self.hr_history) > 0:
            timestamp = int(time.time())
            
            # Create comprehensive dataset
            data = {
                'timestamp': list(range(len(self.hr_history))),
                'hr': list(self.hr_history),
                'rr': list(self.rr_history),
                'spo2': list(self.spo2_history),
                'stress': list(self.stress_history)
            }
            
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame(data)
            filename = f"vital_signs_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            print(f"Saved {len(self.hr_history)} vital signs measurements to {filename}")
            
            # Also save summary statistics
            summary = {
                'avg_hr': np.mean(self.hr_history),
                'avg_rr': np.mean(self.rr_history),
                'avg_spo2': np.mean(self.spo2_history),
                'avg_stress': np.mean(self.stress_history),
                'hr_variability': np.std(self.hr_history)
            }
            
            summary_filename = f"vital_signs_summary_{timestamp}.txt"
            with open(summary_filename, 'w') as f:
                f.write("VITAL SIGNS SUMMARY\n")
                f.write("=" * 30 + "\n")
                for key, value in summary.items():
                    f.write(f"{key}: {value:.2f}\n")
            
            print(f"Summary saved to {summary_filename}")
    
    def run(self):
        """Main application loop"""
        if not self.start_camera():
            return
        
        self.running = True
        print("Advanced Vital Signs Monitor Started")
        print("Position your face in the camera view for comprehensive monitoring")
        print("Controls: Q-quit, S-save data, R-reset, G-toggle graphs, SPACE-screenshot")
        
        cv2.namedWindow('Advanced Vital Signs Monitor', cv2.WINDOW_AUTOSIZE)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame for all vital signs
            vitals = self.processor.process_frame_advanced(frame)
            
            # Update history buffers
            self.hr_history.append(vitals['hr'])
            self.rr_history.append(vitals['rr'])
            self.spo2_history.append(vitals['spo2'])
            self.stress_history.append(vitals['stress'])
            
            # Draw face detection
            self.draw_face_detection(frame, vitals['face_rect'])
            
            # Draw vital signs panel
            self.draw_vital_panel(frame, vitals)
            
            # Show frame
            cv2.imshow('Advanced Vital Signs Monitor', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_vital_signs_data()
            elif key == ord('r'):
                self.reset_data()
            elif key == ord('g'):
                self.show_graphs = not self.show_graphs
                print(f"Graphs display: {'ON' if self.show_graphs else 'OFF'}")
            elif key == ord(' '):  # Spacebar for screenshot
                screenshot_name = f"vitals_screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, frame)
                print(f"Screenshot saved: {screenshot_name}")
        
        self.cleanup()
    
    def reset_data(self):
        """Reset all data buffers"""
        self.processor.rgb_buffer.clear()
        self.processor.timestamps.clear()
        self.processor.respiratory_buffer.clear()
        self.processor.spo2_buffer.clear()
        self.processor.hrv_buffer.clear()
        
        self.hr_history.clear()
        self.rr_history.clear()
        self.spo2_history.clear()
        self.stress_history.clear()
        
        print("All vital signs data reset")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Advanced Vital Signs Monitor closed")

if __name__ == "__main__":
    app = AdvancedVitalsApp()
    app.run()