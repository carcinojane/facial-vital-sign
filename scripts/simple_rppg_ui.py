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

# ITERATION 3: MediaPipe support
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class RPPGProcessor:
    def __init__(self, window_size=300, fps=30, use_multi_roi=True, use_mediapipe=False,
                 use_illumination_norm=False, use_temporal_filter=False, use_motion_detection=False):
        self.window_size = window_size  # 10 seconds at 30fps
        self.fps = fps
        self.rgb_buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # ITERATION 3: MediaPipe or Haar Cascade face detection
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        if self.use_mediapipe:
            self.mp_face = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        else:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.use_multi_roi = use_multi_roi  # ITERATION 2: Multi-ROI improvement

        # ITERATION 4: Signal quality enhancements
        self.use_illumination_norm = use_illumination_norm
        self.use_temporal_filter = use_temporal_filter
        self.use_motion_detection = use_motion_detection

        # For motion detection
        self.previous_frame = None
        self.hr_history = deque(maxlen=10)  # For temporal filtering
        
    def extract_face_roi(self, frame):
        """Extract face ROI using MediaPipe or Haar Cascade

        ITERATION 2 Enhancement: Multi-ROI support
        ITERATION 3 Enhancement: MediaPipe face landmarks for accurate ROI positioning
        """
        if self.use_mediapipe:
            return self._extract_mediapipe_roi(frame)
        else:
            return self._extract_haar_roi(frame)

    def _extract_haar_roi(self, frame):
        """Extract ROI using Haar Cascade (original method)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face

            if self.use_multi_roi:
                # ITERATION 2: Multi-ROI extraction
                # Extract three regions: forehead + left cheek + right cheek
                rois = []

                # Forehead (upper 30% of face, starting 10% from top)
                forehead_y = y + int(h * 0.1)
                forehead_h = int(h * 0.3)
                forehead_roi = frame[forehead_y:forehead_y + forehead_h, x:x + w]
                if forehead_roi.size > 0:
                    rois.append(forehead_roi)

                # Left cheek (40-70% vertical, 0-40% horizontal)
                cheek_y = y + int(h * 0.4)
                cheek_h = int(h * 0.3)
                left_cheek_w = int(w * 0.4)
                left_cheek = frame[cheek_y:cheek_y + cheek_h, x:x + left_cheek_w]
                if left_cheek.size > 0:
                    rois.append(left_cheek)

                # Right cheek (40-70% vertical, 60-100% horizontal)
                right_cheek_x = x + int(w * 0.6)
                right_cheek_w = int(w * 0.4)
                right_cheek = frame[cheek_y:cheek_y + cheek_h, right_cheek_x:right_cheek_x + right_cheek_w]
                if right_cheek.size > 0:
                    rois.append(right_cheek)

                return rois, (x, y, w, h)
            else:
                # BASELINE: Single forehead ROI
                forehead_y = y + int(h * 0.1)
                forehead_h = int(h * 0.4)
                forehead_roi = frame[forehead_y:forehead_y + forehead_h, x:x + w]
                return [forehead_roi], (x, y, w, h)

        return None, None

    def _extract_mediapipe_roi(self, frame):
        """ITERATION 3 (FIXED): Extract ROI using MediaPipe with percentage-based regions

        Uses MediaPipe for accurate face detection, then applies SAME percentage-based
        ROI extraction as Haar Cascade to ensure consistent comparison.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)

        if not results.multi_face_landmarks:
            return None, None

        landmarks = results.multi_face_landmarks[0].landmark

        # Get face bounding box from all landmarks (more accurate than Haar Cascade)
        all_x = [landmarks[i].x for i in range(len(landmarks))]
        all_y = [landmarks[i].y for i in range(len(landmarks))]

        face_x_min = int(min(all_x) * w)
        face_x_max = int(max(all_x) * w)
        face_y_min = int(min(all_y) * h)
        face_y_max = int(max(all_y) * h)

        face_w = face_x_max - face_x_min
        face_h = face_y_max - face_y_min

        if face_w <= 0 or face_h <= 0:
            return None, None

        if self.use_multi_roi:
            # ITERATION 3 (FIXED): Multi-ROI with SAME percentages as Haar Cascade
            rois = []

            # Forehead: Upper 30% of face, starting 10% from top (matches Haar)
            forehead_y1 = face_y_min + int(0.1 * face_h)
            forehead_y2 = face_y_min + int(0.4 * face_h)
            forehead_roi = frame[forehead_y1:forehead_y2, face_x_min:face_x_max]
            if forehead_roi.size > 0:
                rois.append(forehead_roi)

            # Left cheek: 40-70% vertical, 0-40% horizontal (matches Haar)
            left_y1 = face_y_min + int(0.4 * face_h)
            left_y2 = face_y_min + int(0.7 * face_h)
            left_x1 = face_x_min
            left_x2 = face_x_min + int(0.4 * face_w)
            left_cheek_roi = frame[left_y1:left_y2, left_x1:left_x2]
            if left_cheek_roi.size > 0:
                rois.append(left_cheek_roi)

            # Right cheek: 40-70% vertical, 60-100% horizontal (matches Haar)
            right_y1 = face_y_min + int(0.4 * face_h)
            right_y2 = face_y_min + int(0.7 * face_h)
            right_x1 = face_x_min + int(0.6 * face_w)
            right_x2 = face_x_max
            right_cheek_roi = frame[right_y1:right_y2, right_x1:right_x2]
            if right_cheek_roi.size > 0:
                rois.append(right_cheek_roi)

            face_rect = (face_x_min, face_y_min, face_w, face_h)
            return rois, face_rect
        else:
            # Single forehead ROI: Upper 40% of face (matches Haar baseline)
            forehead_y1 = face_y_min
            forehead_y2 = face_y_min + int(0.4 * face_h)
            forehead_roi = frame[forehead_y1:forehead_y2, face_x_min:face_x_max]

            if forehead_roi.size > 0:
                face_rect = (face_x_min, face_y_min, face_w, face_h)
                return [forehead_roi], face_rect

        return None, None

    def extract_rgb_signal(self, rois):
        """Extract mean RGB values from ROI(s)

        ITERATION 2 Enhancement: Support multiple ROIs
        ITERATION 4 Enhancement: Illumination normalization
        - Input: List of ROIs (even if just one)
        - Output: Average RGB across all valid ROIs (normalized if enabled)
        """
        if rois is None or len(rois) == 0:
            return None

        rgb_values = []
        for roi in rois:
            if roi is not None and roi.size > 0:
                # Convert BGR to RGB
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                # ITERATION 4: Illumination normalization (chrominance)
                if self.use_illumination_norm:
                    roi_rgb = self._normalize_illumination(roi_rgb)

                # Calculate mean RGB for this ROI
                mean_rgb = np.mean(roi_rgb.reshape(-1, 3), axis=0)
                rgb_values.append(mean_rgb)

        if len(rgb_values) == 0:
            return None

        # ITERATION 2: Average across all ROIs for robustness
        averaged_rgb = np.mean(rgb_values, axis=0)
        return averaged_rgb

    def _normalize_illumination(self, roi_rgb):
        """ITERATION 4: Chrominance-based illumination normalization

        Reduces effect of ambient lighting changes by normalizing to chrominance space
        """
        roi_float = roi_rgb.astype(np.float32)
        # Add small epsilon to avoid division by zero
        rgb_sum = roi_float[:, :, 0] + roi_float[:, :, 1] + roi_float[:, :, 2] + 1e-6

        # Normalized rgb (chrominance)
        r_norm = roi_float[:, :, 0] / rgb_sum
        g_norm = roi_float[:, :, 1] / rgb_sum
        b_norm = roi_float[:, :, 2] / rgb_sum

        # Scale back to 0-255 range for consistency
        normalized = np.stack([r_norm, g_norm, b_norm], axis=2) * 255
        return normalized.astype(np.uint8)
    
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
    
    def _detect_motion(self, current_frame):
        """ITERATION 4: Detect motion using simple frame differencing

        Returns True if significant motion detected (should skip this frame)
        """
        if self.previous_frame is None:
            self.previous_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return False

        # Convert to grayscale
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calculate frame difference
        diff = np.abs(gray_current.astype(np.float32) - self.previous_frame.astype(np.float32))
        mean_diff = np.mean(diff)

        # Update previous frame
        self.previous_frame = gray_current

        # Threshold: if mean difference > 15, consider it significant motion
        # This is conservative - allows normal micro-movements but rejects head turns
        return mean_diff > 15

    def _apply_temporal_filter(self, hr):
        """ITERATION 4: Apply temporal smoothing to HR estimates

        Uses simple moving average with recent HR history
        """
        if hr <= 0:
            return hr

        self.hr_history.append(hr)

        if len(self.hr_history) < 3:
            return hr  # Not enough history yet

        # Simple moving average over last few estimates
        smoothed_hr = np.mean(list(self.hr_history))
        return smoothed_hr

    def process_frame(self, frame):
        """Process single frame and update buffers

        ITERATION 4 Enhancement: Motion detection and temporal filtering
        """
        # ITERATION 4: Check for motion artifacts
        if self.use_motion_detection and self._detect_motion(frame):
            # Skip this frame due to excessive motion
            return 0.0, None, np.array([])

        roi, face_rect = self.extract_face_roi(frame)

        if roi is not None:
            rgb_signal = self.extract_rgb_signal(roi)
            if rgb_signal is not None:
                self.rgb_buffer.append(rgb_signal)
                self.timestamps.append(time.time())

                # Calculate heart rate if we have enough data
                if len(self.rgb_buffer) >= 60:  # 2 seconds of data
                    hr, filtered_signal = self.pos_algorithm(list(self.rgb_buffer))

                    # ITERATION 4: Apply temporal filtering
                    if self.use_temporal_filter:
                        hr = self._apply_temporal_filter(hr)

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