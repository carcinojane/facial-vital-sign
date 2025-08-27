"""
Simple rPPG UI for testing face scanning and vital sign detection
Uses OpenCV for face detection (instead of MediaPipe) and POS algorithm
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import time
from collections import deque
import threading

class RPPGProcessor:
    def __init__(self, window_size=300, fps=30):
        self.window_size = window_size  # 10 seconds at 30fps
        self.fps = fps
        self.rgb_buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def extract_face_roi(self, frame):
        """Extract face ROI using OpenCV Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract forehead region (upper 40% of face)
            forehead_y = y + int(h * 0.1)
            forehead_h = int(h * 0.4)
            forehead_roi = frame[forehead_y:forehead_y + forehead_h, x:x + w]
            
            return forehead_roi, (x, y, w, h)
        return None, None
    
    def extract_rgb_signal(self, roi):
        """Extract mean RGB values from ROI"""
        if roi is None or roi.size == 0:
            return None
        
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Calculate mean RGB values
        mean_rgb = np.mean(roi_rgb.reshape(-1, 3), axis=0)
        return mean_rgb
    
    def pos_algorithm(self, rgb_signals):
        """
        Plane-Orthogonal-to-Skin (POS) algorithm for rPPG
        Wang et al. "Algorithmic Principles of Remote PPG" (2017)
        """
        if len(rgb_signals) < 30:  # Need at least 1 second of data
            return 0.0, np.array([])
        
        rgb_array = np.array(rgb_signals)
        
        # Normalize signals
        normalized = np.zeros_like(rgb_array)
        for i in range(3):
            channel = rgb_array[:, i]
            normalized[:, i] = (channel - np.mean(channel)) / np.std(channel)
        
        # POS algorithm
        X = normalized
        Xf = np.array([
            X[:, 0] - X[:, 1],  # R - G
            X[:, 0] + X[:, 1] - 2 * X[:, 2]  # R + G - 2B
        ]).T
        
        # Calculate alpha (projection coefficient)
        alpha = np.std(Xf[:, 0]) / np.std(Xf[:, 1])
        
        # POS signal
        pos_signal = Xf[:, 0] - alpha * Xf[:, 1]
        
        # Bandpass filter (0.7 - 3.0 Hz for heart rate)
        sos = signal.butter(4, [0.7, 3.0], btype='band', fs=self.fps, output='sos')
        filtered_signal = signal.sosfilt(sos, pos_signal)
        
        # Heart rate estimation via FFT
        hr = self.estimate_heart_rate(filtered_signal)
        
        return hr, filtered_signal
    
    def estimate_heart_rate(self, signal_data):
        """Estimate heart rate from filtered signal using FFT"""
        if len(signal_data) < 60:  # Need at least 2 seconds
            return 0.0
        
        # FFT
        fft_signal = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.fps)
        
        # Focus on heart rate frequency range (0.7-3.0 Hz = 42-180 BPM)
        valid_idx = (freqs >= 0.7) & (freqs <= 3.0)
        valid_freqs = freqs[valid_idx]
        valid_fft = np.abs(fft_signal[valid_idx])
        
        if len(valid_freqs) > 0:
            # Find peak frequency
            peak_idx = np.argmax(valid_fft)
            peak_freq = valid_freqs[peak_idx]
            hr = peak_freq * 60  # Convert Hz to BPM
            return hr
        
        return 0.0
    
    def process_frame(self, frame):
        """Process single frame and update buffers"""
        roi, face_rect = self.extract_face_roi(frame)
        
        if roi is not None:
            rgb_signal = self.extract_rgb_signal(roi)
            if rgb_signal is not None:
                self.rgb_buffer.append(rgb_signal)
                self.timestamps.append(time.time())
                
                # Calculate heart rate if we have enough data
                if len(self.rgb_buffer) >= 60:  # 2 seconds of data
                    hr, filtered_signal = self.pos_algorithm(list(self.rgb_buffer))
                    return hr, face_rect, filtered_signal
        
        return 0.0, face_rect, np.array([])

class RPPGApp:
    def __init__(self):
        self.processor = RPPGProcessor()
        self.cap = None
        self.running = False
        self.current_hr = 0.0
        self.hr_history = deque(maxlen=100)
        
    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def run(self):
        """Main application loop"""
        if not self.start_camera():
            return
        
        self.running = True
        print("rPPG App Started - Press 'q' to quit, 's' to save HR data")
        
        # Create windows
        cv2.namedWindow('rPPG Scanner', cv2.WINDOW_AUTOSIZE)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            hr, face_rect, filtered_signal = self.processor.process_frame(frame)
            
            if hr > 0:
                self.current_hr = hr
                self.hr_history.append(hr)
            
            # Draw interface
            self.draw_interface(frame, face_rect)
            
            # Show frame
            cv2.imshow('rPPG Scanner', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_hr_data()
            elif key == ord('r'):
                self.reset_data()
        
        self.cleanup()
    
    def draw_interface(self, frame, face_rect):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Draw face rectangle
        if face_rect is not None:
            x, y, w_face, h_face = face_rect
            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
            
            # Draw forehead ROI
            forehead_y = y + int(h_face * 0.1)
            forehead_h = int(h_face * 0.4)
            cv2.rectangle(frame, (x, forehead_y), (x + w_face, forehead_y + forehead_h), (255, 0, 0), 2)
            cv2.putText(frame, "ROI", (x, forehead_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw info panel
        panel_h = 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # HR display
        hr_text = f"Heart Rate: {self.current_hr:.1f} BPM"
        cv2.putText(frame, hr_text, (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        # Statistics
        if len(self.hr_history) > 0:
            avg_hr = np.mean(self.hr_history)
            std_hr = np.std(self.hr_history)
            cv2.putText(frame, f"Avg: {avg_hr:.1f} BPM", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Std: {std_hr:.1f} BPM", (200, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Controls: Q-Quit, S-Save, R-Reset", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Data status
        buffer_status = f"Buffer: {len(self.processor.rgb_buffer)}/{self.processor.window_size}"
        cv2.putText(frame, buffer_status, (w - 200, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def save_hr_data(self):
        """Save heart rate data to file"""
        if len(self.hr_history) > 0:
            timestamp = int(time.time())
            filename = f"hr_data_{timestamp}.txt"
            np.savetxt(filename, list(self.hr_history), fmt='%.2f')
            print(f"Saved {len(self.hr_history)} HR measurements to {filename}")
    
    def reset_data(self):
        """Reset all data buffers"""
        self.processor.rgb_buffer.clear()
        self.processor.timestamps.clear()
        self.hr_history.clear()
        self.current_hr = 0.0
        print("Data buffers reset")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("App closed")

if __name__ == "__main__":
    app = RPPGApp()
    app.run()