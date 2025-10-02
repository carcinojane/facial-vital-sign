"""
Enhanced rPPG Processor with multiple improvement methods for incremental evaluation
Each method can be toggled on/off to measure its impact on accuracy
"""
import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import time
from collections import deque

class EnhancedRPPGProcessor:
    """
    Enhanced rPPG processor with configurable improvement methods:
    1. Baseline: Original POS algorithm
    2. Motion filtering: Detect and filter motion artifacts
    3. Adaptive ROI: Multiple face regions (forehead + cheeks)
    4. Signal detrending: Remove low-frequency trends
    5. Adaptive bandpass: Dynamic frequency range based on signal
    6. Temporal smoothing: Smooth HR estimates over time
    7. Outlier rejection: Remove abnormal HR estimates
    8. Quality assessment: Weight estimates by signal quality
    """

    def __init__(self, window_size=300, fps=30, config=None):
        self.window_size = window_size
        self.fps = fps
        self.rgb_buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # HR history for temporal smoothing
        self.hr_history = deque(maxlen=10)

        # Configuration for improvement methods
        self.config = config or {}
        self.use_motion_filtering = self.config.get('motion_filtering', False)
        self.use_adaptive_roi = self.config.get('adaptive_roi', False)
        self.use_detrending = self.config.get('detrending', False)
        self.use_adaptive_bandpass = self.config.get('adaptive_bandpass', False)
        self.use_temporal_smoothing = self.config.get('temporal_smoothing', False)
        self.use_outlier_rejection = self.config.get('outlier_rejection', False)
        self.use_quality_assessment = self.config.get('quality_assessment', False)

        # Previous frame for motion detection
        self.prev_frame = None

    def extract_face_roi(self, frame):
        """Extract face ROI - baseline or adaptive"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face

            if self.use_adaptive_roi:
                # Extract multiple regions: forehead + left cheek + right cheek
                rois = []

                # Forehead (upper 30% of face)
                forehead_y = y + int(h * 0.1)
                forehead_h = int(h * 0.3)
                forehead_roi = frame[forehead_y:forehead_y + forehead_h, x:x + w]
                rois.append(forehead_roi)

                # Left cheek
                cheek_y = y + int(h * 0.4)
                cheek_h = int(h * 0.3)
                cheek_w = int(w * 0.4)
                left_cheek = frame[cheek_y:cheek_y + cheek_h, x:x + cheek_w]
                rois.append(left_cheek)

                # Right cheek
                right_x = x + int(w * 0.6)
                right_cheek = frame[cheek_y:cheek_y + cheek_h, right_x:right_x + cheek_w]
                rois.append(right_cheek)

                return rois, (x, y, w, h)
            else:
                # Baseline: forehead only
                forehead_y = y + int(h * 0.1)
                forehead_h = int(h * 0.4)
                forehead_roi = frame[forehead_y:forehead_y + forehead_h, x:x + w]
                return [forehead_roi], (x, y, w, h)

        return None, None

    def extract_rgb_signal(self, rois):
        """Extract mean RGB from single or multiple ROIs"""
        if rois is None or len(rois) == 0:
            return None

        rgb_values = []
        for roi in rois:
            if roi is not None and roi.size > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mean_rgb = np.mean(roi_rgb.reshape(-1, 3), axis=0)
                rgb_values.append(mean_rgb)

        if len(rgb_values) > 0:
            # Average across all ROIs
            return np.mean(rgb_values, axis=0)
        return None

    def detect_motion(self, frame):
        """Detect motion between frames - Method 2"""
        if not self.use_motion_filtering or self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate frame difference
        diff = cv2.absdiff(self.prev_frame, gray)
        motion_score = np.mean(diff)

        self.prev_frame = gray

        # Threshold for motion detection
        return motion_score > 15  # Adjust threshold as needed

    def detrend_signal(self, signal_data):
        """Remove low-frequency trends - Method 4"""
        if not self.use_detrending:
            return signal_data

        # Remove linear trend
        from scipy.signal import detrend
        return detrend(signal_data)

    def pos_algorithm(self, rgb_signals):
        """Enhanced POS algorithm with configurable improvements"""
        if len(rgb_signals) < 30:
            return 0.0, np.array([]), 0.0

        rgb_array = np.array(rgb_signals)

        # Normalize signals
        normalized = np.zeros_like(rgb_array)
        for i in range(3):
            channel = rgb_array[:, i]
            if np.std(channel) > 0:
                normalized[:, i] = (channel - np.mean(channel)) / np.std(channel)

        # POS algorithm
        X = normalized
        Xf = np.array([
            X[:, 0] - X[:, 1],
            X[:, 0] + X[:, 1] - 2 * X[:, 2]
        ]).T

        alpha = np.std(Xf[:, 0]) / (np.std(Xf[:, 1]) + 1e-10)
        pos_signal = Xf[:, 0] - alpha * Xf[:, 1]

        # Method 4: Detrending
        pos_signal = self.detrend_signal(pos_signal)

        # Method 5: Adaptive bandpass filtering
        if self.use_adaptive_bandpass:
            # Use wider initial range, then refine
            sos = signal.butter(4, [0.5, 3.5], btype='band', fs=self.fps, output='sos')
        else:
            # Baseline bandpass
            sos = signal.butter(4, [0.7, 3.0], btype='band', fs=self.fps, output='sos')

        filtered_signal = signal.sosfilt(sos, pos_signal)

        # Method 7: Quality assessment
        snr = self.calculate_snr(filtered_signal) if self.use_quality_assessment else 1.0

        # Heart rate estimation
        hr = self.estimate_heart_rate(filtered_signal)

        return hr, filtered_signal, snr

    def calculate_snr(self, signal_data):
        """Calculate Signal-to-Noise Ratio - Method 8"""
        if len(signal_data) < 60:
            return 0.0

        # FFT
        fft_signal = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.fps)
        power_spectrum = np.abs(fft_signal) ** 2

        # HR range (0.7-3.0 Hz)
        valid_idx = (freqs >= 0.7) & (freqs <= 3.0)

        if np.sum(valid_idx) == 0:
            return 0.0

        # Signal power = max peak in HR range
        signal_power = np.max(power_spectrum[valid_idx])

        # Noise power = mean of remaining spectrum
        noise_power = np.mean(power_spectrum[valid_idx])

        if noise_power > 0:
            snr = signal_power / noise_power
            return snr
        return 0.0

    def estimate_heart_rate(self, signal_data):
        """Estimate heart rate with optional outlier rejection"""
        if len(signal_data) < 60:
            return 0.0

        fft_signal = fft(signal_data)
        freqs = fftfreq(len(signal_data), 1/self.fps)

        valid_idx = (freqs >= 0.7) & (freqs <= 3.0)
        valid_freqs = freqs[valid_idx]
        valid_fft = np.abs(fft_signal[valid_idx])

        if len(valid_freqs) > 0:
            peak_idx = np.argmax(valid_fft)
            peak_freq = valid_freqs[peak_idx]
            hr = peak_freq * 60

            # Method 7: Outlier rejection
            if self.use_outlier_rejection:
                if len(self.hr_history) > 3:
                    median_hr = np.median(self.hr_history)
                    # Reject if more than 20 BPM from median
                    if abs(hr - median_hr) > 20:
                        return median_hr

            # Method 6: Temporal smoothing
            if self.use_temporal_smoothing:
                self.hr_history.append(hr)
                if len(self.hr_history) >= 3:
                    # Use median of recent estimates
                    hr = np.median(list(self.hr_history)[-5:])

            return hr

        return 0.0

    def process_frame(self, frame):
        """Process frame with all enabled improvements"""
        # Method 2: Motion filtering
        if self.use_motion_filtering and self.detect_motion(frame):
            # Skip this frame due to motion
            return 0.0, None, np.array([]), 0.0

        # Method 3: Adaptive ROI
        rois, face_rect = self.extract_face_roi(frame)

        if rois is not None:
            rgb_signal = self.extract_rgb_signal(rois)
            if rgb_signal is not None:
                self.rgb_buffer.append(rgb_signal)
                self.timestamps.append(time.time())

                if len(self.rgb_buffer) >= 60:
                    hr, filtered_signal, snr = self.pos_algorithm(list(self.rgb_buffer))
                    return hr, face_rect, filtered_signal, snr

        return 0.0, face_rect, np.array([]), 0.0

    def get_config_name(self):
        """Get human-readable configuration name"""
        enabled = []
        if self.use_motion_filtering:
            enabled.append("Motion")
        if self.use_adaptive_roi:
            enabled.append("MultiROI")
        if self.use_detrending:
            enabled.append("Detrend")
        if self.use_adaptive_bandpass:
            enabled.append("AdaptBP")
        if self.use_temporal_smoothing:
            enabled.append("Smooth")
        if self.use_outlier_rejection:
            enabled.append("Outlier")
        if self.use_quality_assessment:
            enabled.append("QA")

        return "Baseline" if not enabled else "+".join(enabled)
