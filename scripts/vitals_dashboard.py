"""
Real-time Vitals Dashboard with graphs and comprehensive monitoring
Displays multiple vital signs with live plotting capabilities
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import threading
import time
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.advanced_vitals_ui import AdvancedVitalsProcessor

class VitalsDashboard:
    def __init__(self):
        self.processor = AdvancedVitalsProcessor()
        self.cap = None
        self.running = False
        
        # Data storage with timestamps
        self.timestamps = deque(maxlen=300)
        self.hr_data = deque(maxlen=300)
        self.rr_data = deque(maxlen=300)
        self.spo2_data = deque(maxlen=300)
        self.stress_data = deque(maxlen=300)
        self.hrv_data = deque(maxlen=300)
        self.quality_data = deque(maxlen=300)
        self.bp_systolic_data = deque(maxlen=300)
        self.bp_diastolic_data = deque(maxlen=300)
        
        # Dashboard settings
        self.show_dashboard = False
        self.dashboard_thread = None
        self.current_vitals = {}
        
        # Statistics
        self.session_start_time = None
        self.total_measurements = 0
        
    def start_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        return True
    
    def update_data(self, vitals):
        """Update data buffers with new measurements"""
        current_time = datetime.now()
        
        self.timestamps.append(current_time)
        self.hr_data.append(vitals.get('hr', 0))
        self.rr_data.append(vitals.get('rr', 0))
        self.spo2_data.append(vitals.get('spo2', 95))
        self.stress_data.append(vitals.get('stress', 0))
        self.hrv_data.append(vitals.get('hrv', 0))
        self.quality_data.append(vitals.get('signal_quality', 0))
        self.bp_systolic_data.append(vitals.get('bp_systolic', 120))
        self.bp_diastolic_data.append(vitals.get('bp_diastolic', 80))
        
        self.current_vitals = vitals.copy()
        self.total_measurements += 1
    
    def get_vital_status(self, vital_type, value):
        """Get vital sign status and color"""
        if vital_type == 'hr':
            if 60 <= value <= 100:
                return 'Normal', (0, 255, 0)
            elif 50 <= value <= 110:
                return 'Acceptable', (0, 255, 255)
            elif value > 110:
                return 'High', (0, 165, 255)
            else:
                return 'Low', (0, 0, 255)
                
        elif vital_type == 'rr':
            if 12 <= value <= 20:
                return 'Normal', (0, 255, 0)
            elif 8 <= value <= 25:
                return 'Acceptable', (0, 255, 255)
            else:
                return 'Abnormal', (0, 0, 255)
                
        elif vital_type == 'spo2':
            if value >= 98:
                return 'Excellent', (0, 255, 0)
            elif value >= 95:
                return 'Normal', (0, 255, 255)
            elif value >= 90:
                return 'Low', (0, 165, 255)
            else:
                return 'Critical', (0, 0, 139)
                
        elif vital_type == 'stress':
            if value <= 25:
                return 'Low', (0, 255, 0)
            elif value <= 50:
                return 'Moderate', (0, 255, 255)
            elif value <= 75:
                return 'High', (0, 165, 255)
            else:
                return 'Very High', (0, 0, 255)
        
        return 'Unknown', (128, 128, 128)
    
    def draw_enhanced_ui(self, frame):
        """Draw enhanced UI with comprehensive vital signs display"""
        h, w = frame.shape[:2]
        
        if not self.current_vitals:
            # No data yet
            cv2.putText(frame, "Initializing vital signs monitoring...", 
                       (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame
        
        # Create dark overlay for better readability
        overlay = frame.copy()
        
        # Top status bar
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        
        # Main vital signs panel (larger)
        panel_height = 250
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        
        # Apply overlay
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        # Draw status bar
        self.draw_status_bar(frame, w)
        
        # Draw main vitals panel
        self.draw_main_vitals_panel(frame, h, w, panel_height)
        
        # Draw face detection if present
        if self.current_vitals.get('face_rect'):
            self.draw_face_indicators(frame, self.current_vitals['face_rect'])
        
        return frame
    
    def draw_status_bar(self, frame, width):
        """Draw top status bar with session info"""
        # Session time
        if self.session_start_time:
            elapsed = datetime.now() - self.session_start_time
            elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        else:
            elapsed_str = "00:00:00"
        
        cv2.putText(frame, f"Session: {elapsed_str}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Measurements count
        cv2.putText(frame, f"Measurements: {self.total_measurements}", (300, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (width - 180, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Dashboard status
        dashboard_status = "ON" if self.show_dashboard else "OFF"
        cv2.putText(frame, f"Dashboard: {dashboard_status}", (width - 180, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def draw_main_vitals_panel(self, frame, height, width, panel_height):
        """Draw comprehensive vitals panel"""
        vitals = self.current_vitals
        panel_top = height - panel_height + 20
        
        # Column 1: Heart Rate (prominent)
        col1_x = 30
        hr_status, hr_color = self.get_vital_status('hr', vitals.get('hr', 0))
        
        cv2.putText(frame, "HEART RATE", (col1_x, panel_top), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"{vitals.get('hr', 0):.0f}", (col1_x, panel_top + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, hr_color, 3)
        cv2.putText(frame, "BPM", (col1_x, panel_top + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2)
        cv2.putText(frame, hr_status, (col1_x, panel_top + 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hr_color, 2)
        
        # HRV below HR
        cv2.putText(frame, f"HRV: {vitals.get('hrv', 0):.1f} ms", (col1_x, panel_top + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Column 2: Respiratory Rate
        col2_x = 280
        rr_status, rr_color = self.get_vital_status('rr', vitals.get('rr', 0))
        
        cv2.putText(frame, "RESPIRATORY RATE", (col2_x, panel_top), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"{vitals.get('rr', 0):.0f}", (col2_x, panel_top + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, rr_color, 3)
        cv2.putText(frame, "BrPM", (col2_x, panel_top + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, rr_color, 2)
        cv2.putText(frame, rr_status, (col2_x, panel_top + 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, rr_color, 2)
        
        # Column 3: SpO2
        col3_x = 480
        spo2_status, spo2_color = self.get_vital_status('spo2', vitals.get('spo2', 95))
        
        cv2.putText(frame, "OXYGEN SATURATION", (col3_x, panel_top), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"{vitals.get('spo2', 95):.0f}%", (col3_x, panel_top + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, spo2_color, 3)
        cv2.putText(frame, "SpO2", (col3_x, panel_top + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, spo2_color, 2)
        cv2.putText(frame, spo2_status, (col3_x, panel_top + 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, spo2_color, 2)
        
        # Column 4: Stress Level
        col4_x = 680
        stress_status, stress_color = self.get_vital_status('stress', vitals.get('stress', 0))
        
        cv2.putText(frame, "STRESS LEVEL", (col4_x, panel_top), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"{vitals.get('stress', 0):.0f}%", (col4_x, panel_top + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, stress_color, 3)
        cv2.putText(frame, stress_status, (col4_x, panel_top + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, stress_color, 2)
        
        # Stress level progress bar
        bar_x, bar_y = col4_x, panel_top + 90
        bar_width, bar_height = 120, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 2)
        stress_fill = int((vitals.get('stress', 0) / 100) * bar_width)
        if stress_fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + stress_fill, bar_y + bar_height), stress_color, -1)
        
        # Column 5: Signal Quality
        col5_x = 900
        quality = vitals.get('signal_quality', 0)
        quality_color = (0, 255, 0) if quality > 80 else (0, 255, 255) if quality > 60 else (0, 0, 255)
        
        cv2.putText(frame, "SIGNAL QUALITY", (col5_x, panel_top), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"{quality:.0f}%", (col5_x, panel_top + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, quality_color, 3)
        
        # Quality progress bar
        bar_x, bar_y = col5_x, panel_top + 70
        bar_width, bar_height = 100, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 1)
        quality_fill = int((quality / 100) * bar_width)
        if quality_fill > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + quality_fill, bar_y + bar_height), quality_color, -1)
        
        # Statistics summary
        if len(self.hr_data) > 10:
            stats_y = panel_top + 140
            cv2.putText(frame, f"Avg HR: {np.mean(list(self.hr_data)[-30:]):.0f}  " +
                             f"Avg RR: {np.mean(list(self.rr_data)[-30:]):.0f}  " +
                             f"Avg SpO2: {np.mean(list(self.spo2_data)[-30:]):.0f}%", 
                       (30, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        
        # Controls
        controls_y = height - 25
        cv2.putText(frame, "Controls: Q-Quit | S-Save Data | R-Reset | D-Dashboard | SPACE-Screenshot | P-Pause", 
                   (30, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def draw_face_indicators(self, frame, face_rect):
        """Draw enhanced face detection indicators"""
        if face_rect is None:
            return
        
        x, y, w, h = face_rect
        
        # Main face rectangle with rounded corners effect
        cv2.rectangle(frame, (x-2, y-2), (x + w + 2, y + h + 2), (0, 255, 0), 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        # Face detection confidence indicator
        cv2.putText(frame, "FACE DETECTED", (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # ROI indicators
        # Forehead ROI (HR)
        forehead_y = y + int(h * 0.1)
        forehead_h = int(h * 0.4)
        cv2.rectangle(frame, (x, forehead_y), (x + w, forehead_y + forehead_h), (255, 100, 0), 2)
        cv2.putText(frame, "HR", (x + 5, forehead_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        
        # Cheek ROI (RR)
        cheek_y = y + int(h * 0.3)
        cheek_h = int(h * 0.4)
        cheek_w = int(w * 0.3)
        cv2.rectangle(frame, (x + int(w * 0.1), cheek_y), 
                     (x + int(w * 0.1) + cheek_w, cheek_y + cheek_h), (0, 255, 255), 2)
        cv2.putText(frame, "RR", (x + int(w * 0.1) + 5, cheek_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def create_dashboard_window(self):
        """Create matplotlib dashboard in separate thread"""
        if self.show_dashboard:
            # Setup matplotlib for real-time plotting
            plt.ion()  # Interactive mode
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Real-time Vital Signs Dashboard', fontsize=16)
            
            # Configure subplots
            axes[0, 0].set_title('Heart Rate (BPM)')
            axes[0, 0].set_ylim(40, 140)
            
            axes[0, 1].set_title('Respiratory Rate (BrPM)')  
            axes[0, 1].set_ylim(5, 35)
            
            axes[0, 2].set_title('SpO2 (%)')
            axes[0, 2].set_ylim(85, 100)
            
            axes[1, 0].set_title('Stress Level (%)')
            axes[1, 0].set_ylim(0, 100)
            
            axes[1, 1].set_title('HRV (RMSSD)')
            axes[1, 1].set_ylim(0, 100)
            
            axes[1, 2].set_title('Signal Quality (%)')
            axes[1, 2].set_ylim(0, 100)
            
            plt.tight_layout()
            
            # Real-time update loop
            while self.show_dashboard and self.running:
                try:
                    if len(self.timestamps) > 1:
                        times = list(self.timestamps)
                        
                        # Clear and plot each vital sign
                        axes[0, 0].clear()
                        axes[0, 0].plot(times, list(self.hr_data), 'r-', linewidth=2)
                        axes[0, 0].set_title('Heart Rate (BPM)')
                        axes[0, 0].set_ylim(40, 140)
                        axes[0, 0].grid(True)
                        
                        axes[0, 1].clear()
                        axes[0, 1].plot(times, list(self.rr_data), 'b-', linewidth=2)
                        axes[0, 1].set_title('Respiratory Rate (BrPM)')
                        axes[0, 1].set_ylim(5, 35)
                        axes[0, 1].grid(True)
                        
                        axes[0, 2].clear()
                        axes[0, 2].plot(times, list(self.spo2_data), 'g-', linewidth=2)
                        axes[0, 2].set_title('SpO2 (%)')
                        axes[0, 2].set_ylim(85, 100)
                        axes[0, 2].grid(True)
                        
                        axes[1, 0].clear()
                        axes[1, 0].plot(times, list(self.stress_data), 'm-', linewidth=2)
                        axes[1, 0].set_title('Stress Level (%)')
                        axes[1, 0].set_ylim(0, 100)
                        axes[1, 0].grid(True)
                        
                        axes[1, 1].clear()
                        axes[1, 1].plot(times, list(self.hrv_data), 'c-', linewidth=2)
                        axes[1, 1].set_title('HRV (RMSSD)')
                        axes[1, 1].set_ylim(0, 100)
                        axes[1, 1].grid(True)
                        
                        axes[1, 2].clear()
                        axes[1, 2].plot(times, list(self.quality_data), 'orange', linewidth=2)
                        axes[1, 2].set_title('Signal Quality (%)')
                        axes[1, 2].set_ylim(0, 100)
                        axes[1, 2].grid(True)
                        
                        # Format x-axis to show time
                        for ax in axes.flat:
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
                            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                        
                        plt.tight_layout()
                        plt.pause(0.1)  # Small pause for smooth updates
                    
                    time.sleep(0.5)  # Update every 500ms
                    
                except Exception as e:
                    print(f"Dashboard error: {e}")
                    break
            
            plt.close(fig)
    
    def toggle_dashboard(self):
        """Toggle the matplotlib dashboard"""
        self.show_dashboard = not self.show_dashboard
        
        if self.show_dashboard:
            print("Starting dashboard...")
            self.dashboard_thread = threading.Thread(target=self.create_dashboard_window)
            self.dashboard_thread.daemon = True
            self.dashboard_thread.start()
        else:
            print("Stopping dashboard...")
            if self.dashboard_thread:
                self.dashboard_thread.join(timeout=1)
    
    def save_session_data(self):
        """Save comprehensive session data"""
        if not self.timestamps:
            print("No data to save")
            return
        
        timestamp = int(time.time())
        
        # Create comprehensive dataset
        data = []
        for i, ts in enumerate(self.timestamps):
            data.append({
                'timestamp': ts.isoformat(),
                'hr': self.hr_data[i] if i < len(self.hr_data) else 0,
                'rr': self.rr_data[i] if i < len(self.rr_data) else 0,
                'spo2': self.spo2_data[i] if i < len(self.spo2_data) else 0,
                'stress': self.stress_data[i] if i < len(self.stress_data) else 0,
                'hrv': self.hrv_data[i] if i < len(self.hrv_data) else 0,
                'quality': self.quality_data[i] if i < len(self.quality_data) else 0
            })
        
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame(data)
        filename = f"vitals_session_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Generate summary report
        summary_filename = f"vitals_summary_{timestamp}.txt"
        self.generate_summary_report(summary_filename)
        
        print(f"Session data saved:")
        print(f"  Data file: {filename} ({len(data)} measurements)")
        print(f"  Summary: {summary_filename}")
    
    def generate_summary_report(self, filename):
        """Generate comprehensive summary report"""
        if not self.hr_data:
            return
        
        with open(filename, 'w') as f:
            f.write("VITAL SIGNS MONITORING SESSION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Session info
            f.write("SESSION INFORMATION:\n")
            f.write("-" * 30 + "\n")
            if self.session_start_time:
                elapsed = datetime.now() - self.session_start_time
                f.write(f"Duration: {elapsed}\n")
            f.write(f"Total measurements: {self.total_measurements}\n")
            f.write(f"Data points: {len(self.hr_data)}\n\n")
            
            # Vital signs statistics
            f.write("VITAL SIGNS STATISTICS:\n")
            f.write("-" * 30 + "\n")
            
            # Heart Rate
            hr_values = [x for x in self.hr_data if x > 0]
            if hr_values:
                f.write(f"Heart Rate (BPM):\n")
                f.write(f"  Average: {np.mean(hr_values):.1f}\n")
                f.write(f"  Range: {np.min(hr_values):.1f} - {np.max(hr_values):.1f}\n")
                f.write(f"  Std Dev: {np.std(hr_values):.1f}\n\n")
            
            # Respiratory Rate
            rr_values = [x for x in self.rr_data if x > 0]
            if rr_values:
                f.write(f"Respiratory Rate (BrPM):\n")
                f.write(f"  Average: {np.mean(rr_values):.1f}\n")
                f.write(f"  Range: {np.min(rr_values):.1f} - {np.max(rr_values):.1f}\n")
                f.write(f"  Std Dev: {np.std(rr_values):.1f}\n\n")
            
            # SpO2
            spo2_values = [x for x in self.spo2_data if x > 0]
            if spo2_values:
                f.write(f"Oxygen Saturation (%):\n")
                f.write(f"  Average: {np.mean(spo2_values):.1f}\n")
                f.write(f"  Range: {np.min(spo2_values):.1f} - {np.max(spo2_values):.1f}\n")
                f.write(f"  Std Dev: {np.std(spo2_values):.1f}\n\n")
            
            # Stress Level
            stress_values = [x for x in self.stress_data if x >= 0]
            if stress_values:
                f.write(f"Stress Level (%):\n")
                f.write(f"  Average: {np.mean(stress_values):.1f}\n")
                f.write(f"  Range: {np.min(stress_values):.1f} - {np.max(stress_values):.1f}\n")
                f.write(f"  Std Dev: {np.std(stress_values):.1f}\n\n")
            
            # Signal Quality
            quality_values = [x for x in self.quality_data if x > 0]
            if quality_values:
                f.write(f"Signal Quality (%):\n")
                f.write(f"  Average: {np.mean(quality_values):.1f}\n")
                f.write(f"  Range: {np.min(quality_values):.1f} - {np.max(quality_values):.1f}\n\n")
    
    def run(self):
        """Main application loop"""
        if not self.start_camera():
            print("Error: Could not start camera")
            return
        
        self.running = True
        self.session_start_time = datetime.now()
        
        print("Advanced Vital Signs Dashboard Started")
        print("Controls:")
        print("  Q - Quit")
        print("  S - Save session data")
        print("  R - Reset all data")
        print("  D - Toggle live dashboard")
        print("  P - Pause/Resume monitoring")
        print("  SPACE - Take screenshot")
        
        cv2.namedWindow('Vital Signs Dashboard', cv2.WINDOW_AUTOSIZE)
        
        paused = False
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if not paused:
                # Process frame for vital signs
                vitals = self.processor.process_frame_advanced(frame)
                
                # Update data buffers
                self.update_data(vitals)
            
            # Draw enhanced UI
            frame = self.draw_enhanced_ui(frame)
            
            # Show frame
            cv2.imshow('Vital Signs Dashboard', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_session_data()
            elif key == ord('r'):
                self.reset_all_data()
            elif key == ord('d'):
                self.toggle_dashboard()
            elif key == ord('p'):
                paused = not paused
                print(f"Monitoring {'paused' if paused else 'resumed'}")
            elif key == ord(' '):  # Screenshot
                screenshot_name = f"dashboard_screenshot_{int(time.time())}.png"
                cv2.imwrite(screenshot_name, frame)
                print(f"Screenshot saved: {screenshot_name}")
        
        # Cleanup
        self.show_dashboard = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("Vital Signs Dashboard closed")
    
    def reset_all_data(self):
        """Reset all data buffers"""
        self.timestamps.clear()
        self.hr_data.clear()
        self.rr_data.clear()
        self.spo2_data.clear()
        self.stress_data.clear()
        self.hrv_data.clear()
        self.quality_data.clear()
        
        # Reset processor buffers
        self.processor.rgb_buffer.clear()
        self.processor.timestamps.clear()
        self.processor.respiratory_buffer.clear()
        self.processor.spo2_buffer.clear()
        self.processor.hrv_buffer.clear()
        
        self.session_start_time = datetime.now()
        self.total_measurements = 0
        
        print("All data reset - new session started")

if __name__ == "__main__":
    dashboard = VitalsDashboard()
    dashboard.run()