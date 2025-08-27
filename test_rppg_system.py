"""
Quick test script for the rPPG system
Tests both UI and evaluation components
"""
import cv2
import numpy as np
from scripts.simple_rppg_ui import RPPGProcessor
import os

def test_opencv_camera():
    """Test if OpenCV can access camera"""
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("FAIL Cannot access camera")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("FAIL Cannot read from camera")
        cap.release()
        return False
    
    print(f"OK Camera working: {frame.shape}")
    cap.release()
    return True

def test_face_detection():
    """Test face detection with OpenCV"""
    print("Testing face detection...")
    
    # Create test processor
    processor = RPPGProcessor()
    
    # Try with webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FAIL Camera not available for face detection test")
        return False
    
    print("Show your face to the camera for 5 seconds...")
    face_detected = False
    
    for i in range(150):  # 5 seconds at ~30fps
        ret, frame = cap.read()
        if not ret:
            continue
            
        roi, face_rect = processor.extract_face_roi(frame)
        
        if face_rect is not None:
            face_detected = True
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if face_detected:
        print("OK Face detection working")
        return True
    else:
        print("FAIL No face detected")
        return False

def test_rppg_processing():
    """Test rPPG signal processing"""
    print("Testing rPPG signal processing...")
    
    # Create synthetic RGB signal (simulate heartbeat)
    fs = 30  # 30 FPS
    duration = 10  # 10 seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Simulate heartbeat at 70 BPM (1.17 Hz)
    hr_freq = 70 / 60  # Convert BPM to Hz
    heartbeat_signal = 0.1 * np.sin(2 * np.pi * hr_freq * t)
    
    # Add to baseline RGB values
    baseline_rgb = [128, 100, 80]  # Typical skin color values
    synthetic_rgb = []
    
    for i in range(len(t)):
        # Heartbeat affects R channel most, G channel less, B channel least
        r = baseline_rgb[0] + heartbeat_signal[i] * 10
        g = baseline_rgb[1] + heartbeat_signal[i] * 5
        b = baseline_rgb[2] + heartbeat_signal[i] * 2
        synthetic_rgb.append([r, g, b])
    
    # Test POS algorithm
    processor = RPPGProcessor(fps=fs)
    
    # Simulate processing frame by frame
    for rgb in synthetic_rgb:
        processor.rgb_buffer.append(np.array(rgb))
    
    # Run POS algorithm
    hr_estimated, filtered_signal = processor.pos_algorithm(list(processor.rgb_buffer))
    
    print(f"Input HR: 70.0 BPM")
    print(f"Estimated HR: {hr_estimated:.1f} BPM")
    print(f"Error: {abs(hr_estimated - 70.0):.1f} BPM")
    
    # Check if estimate is reasonable (within 10 BPM)
    if abs(hr_estimated - 70.0) < 10.0:
        print("OK rPPG processing working")
        return True
    else:
        print("FAIL rPPG processing not working correctly")
        return False

def test_dataset_access():
    """Test dataset access"""
    print("Testing dataset access...")
    
    dataset_path = r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data"
    
    if not os.path.exists(dataset_path):
        print(f"FAIL Dataset path not found: {dataset_path}")
        return False
    
    pure_path = os.path.join(dataset_path, "PURE")
    if not os.path.exists(pure_path):
        print(f"FAIL PURE dataset not found: {pure_path}")
        return False
    
    # Count sessions
    sessions = [d for d in os.listdir(pure_path) if os.path.isdir(os.path.join(pure_path, d))]
    print(f"OK Found {len(sessions)} PURE sessions")
    
    # Check a few sessions for structure
    valid_sessions = 0
    for session in sessions[:3]:  # Check first 3
        session_path = os.path.join(pure_path, session)
        json_files = [f for f in os.listdir(session_path) if f.endswith('.json')]
        video_files = [f for f in os.listdir(session_path) if f.endswith('.avi')]
        
        if json_files and video_files:
            valid_sessions += 1
    
    print(f"OK {valid_sessions}/3 sessions have required files")
    return valid_sessions > 0

def run_quick_ui_test():
    """Run UI for a quick test"""
    print("Starting quick UI test...")
    print("This will open the rPPG interface for 30 seconds")
    print("Position your face in the green rectangle and wait for HR detection")
    
    from scripts.simple_rppg_ui import RPPGApp
    
    # Modify app for quick test
    class QuickTestApp(RPPGApp):
        def __init__(self):
            super().__init__()
            self.test_duration = 30  # 30 seconds
            self.start_time = None
        
        def run(self):
            if not self.start_camera():
                return
            
            import time
            self.start_time = time.time()
            self.running = True
            
            cv2.namedWindow('rPPG Quick Test', cv2.WINDOW_AUTOSIZE)
            
            while self.running:
                if time.time() - self.start_time > self.test_duration:
                    print(f"Test completed! Collected {len(self.hr_history)} HR measurements")
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                hr, face_rect, filtered_signal = self.processor.process_frame(frame)
                
                if hr > 0:
                    self.current_hr = hr
                    self.hr_history.append(hr)
                
                # Draw test interface
                self.draw_test_interface(frame, face_rect)
                
                cv2.imshow('rPPG Quick Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.cleanup()
            
            # Show results
            if len(self.hr_history) > 0:
                avg_hr = np.mean(self.hr_history)
                std_hr = np.std(self.hr_history)
                print(f"Average HR: {avg_hr:.1f} ± {std_hr:.1f} BPM")
                print("✅ UI test completed successfully")
            else:
                print("❌ No HR measurements collected")
        
        def draw_test_interface(self, frame, face_rect):
            """Simplified interface for testing"""
            h, w = frame.shape[:2]
            
            # Draw face rectangle
            if face_rect is not None:
                x, y, w_face, h_face = face_rect
                cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
                cv2.putText(frame, "Face Found", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # HR display
            hr_text = f"HR: {self.current_hr:.1f} BPM"
            cv2.putText(frame, hr_text, (50, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Test info
            remaining = max(0, self.test_duration - (time.time() - self.start_time))
            cv2.putText(frame, f"Test: {remaining:.0f}s left", (50, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {len(self.hr_history)}", (300, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    app = QuickTestApp()
    app.run()

def main():
    """Run all tests"""
    print("=" * 50)
    print("rPPG SYSTEM TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Camera Access", test_opencv_camera),
        ("Dataset Access", test_dataset_access),
        ("rPPG Processing", test_rppg_processing),
        ("Face Detection", test_face_detection),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<20}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests passed! System is ready.")
        
        # Offer to run UI test
        response = input("\nRun quick UI test? (y/n): ")
        if response.lower() == 'y':
            run_quick_ui_test()
    else:
        print("\nSome tests failed. Check the issues above.")

if __name__ == "__main__":
    main()