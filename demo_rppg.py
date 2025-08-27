"""
Quick demo of rPPG system with automatic testing
"""
import cv2
import numpy as np
from scripts.simple_rppg_ui import RPPGProcessor
import time

def demo_with_webcam():
    """Demo rPPG with webcam for 30 seconds"""
    print("Starting rPPG Demo...")
    print("Position your face in the camera view")
    print("Demo will run for 30 seconds")
    
    # Initialize
    processor = RPPGProcessor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot access camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Demo variables
    start_time = time.time()
    frame_count = 0
    hr_readings = []
    
    print("Demo started - press 'q' to quit early")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > 30:  # 30 seconds
            break
        
        # Process frame
        hr, face_rect, filtered_signal = processor.process_frame(frame)
        frame_count += 1
        
        if hr > 0:
            hr_readings.append(hr)
        
        # Draw on frame
        h, w = frame.shape[:2]
        
        # Face detection
        if face_rect is not None:
            x, y, fw, fh = face_rect
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ROI
            forehead_y = y + int(fh * 0.1)
            forehead_h = int(fh * 0.4)
            cv2.rectangle(frame, (x, forehead_y), (x + fw, forehead_y + forehead_h), (255, 0, 0), 2)
        
        # Info display
        cv2.putText(frame, f"HR: {hr:.1f} BPM", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, f"Time: {elapsed:.1f}s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Readings: {len(hr_readings)}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if len(hr_readings) > 0:
            avg_hr = np.mean(hr_readings)
            cv2.putText(frame, f"Avg HR: {avg_hr:.1f} BPM", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('rPPG Demo', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup and results
    cap.release()
    cv2.destroyAllWindows()
    
    # Print results
    print("\n" + "="*50)
    print("DEMO RESULTS")
    print("="*50)
    print(f"Duration: {elapsed:.1f} seconds")
    print(f"Frames processed: {frame_count}")
    print(f"FPS: {frame_count/elapsed:.1f}")
    print(f"HR readings: {len(hr_readings)}")
    
    if len(hr_readings) > 0:
        avg_hr = np.mean(hr_readings)
        std_hr = np.std(hr_readings)
        min_hr = np.min(hr_readings)
        max_hr = np.max(hr_readings)
        
        print(f"Average HR: {avg_hr:.1f} ± {std_hr:.1f} BPM")
        print(f"HR Range: {min_hr:.1f} - {max_hr:.1f} BPM")
        
        # Check if readings are reasonable
        if 50 <= avg_hr <= 120:
            print("OK HR readings appear reasonable")
        else:
            print("WARNING HR readings may need calibration")
        
        # Save data
        timestamp = int(time.time())
        filename = f"demo_hr_data_{timestamp}.txt"
        np.savetxt(filename, hr_readings, fmt='%.2f')
        print(f"HR data saved to: {filename}")
        
        # Simple performance assessment
        if len(hr_readings) >= 10:  # Need at least 10 readings
            stability = std_hr / avg_hr * 100  # CV%
            print(f"Stability (CV): {stability:.1f}%")
            
            if stability < 10:
                print("OK Good stability")
            elif stability < 20:
                print("OK Moderate stability")
            else:
                print("WARNING Poor stability - check lighting/movement")
    else:
        print("FAIL No HR readings obtained")
        print("Try:")
        print("- Better lighting")
        print("- Sit closer to camera")
        print("- Reduce movement")
        print("- Check face detection")
    
    print("\nDemo completed!")

def test_synthetic_data():
    """Test with synthetic data to verify algorithm"""
    print("\nTesting algorithm with synthetic data...")
    
    processor = RPPGProcessor(fps=30)
    
    # Create 10 seconds of synthetic data at 72 BPM
    fs = 30
    duration = 10
    t = np.linspace(0, duration, fs * duration)
    hr_actual = 72  # BPM
    freq_actual = hr_actual / 60  # Hz
    
    # Simulate PPG signal in RGB channels
    ppg_signal = 0.05 * np.sin(2 * np.pi * freq_actual * t)
    noise = 0.02 * np.random.randn(len(t))
    
    baseline = [120, 80, 60]  # Typical skin RGB
    synthetic_data = []
    
    for i in range(len(t)):
        # PPG affects red channel most
        r = baseline[0] + ppg_signal[i] * 100 + noise[i] * 50
        g = baseline[1] + ppg_signal[i] * 30 + noise[i] * 30
        b = baseline[2] + ppg_signal[i] * 10 + noise[i] * 20
        synthetic_data.append([r, g, b])
    
    # Add to processor buffer
    for rgb in synthetic_data:
        processor.rgb_buffer.append(np.array(rgb))
    
    # Test algorithm
    hr_estimated, signal = processor.pos_algorithm(list(processor.rgb_buffer))
    
    error = abs(hr_estimated - hr_actual)
    print(f"Synthetic test: {hr_actual} BPM -> {hr_estimated:.1f} BPM (error: {error:.1f})")
    
    if error < 10:
        print("OK Algorithm working correctly")
        return True
    else:
        print("WARNING Algorithm may need tuning")
        return False

def main():
    print("rPPG SYSTEM DEMO")
    print("="*50)
    
    # Test algorithm first
    algo_ok = test_synthetic_data()
    
    if algo_ok:
        response = input("\nRun webcam demo? (y/n): ")
        if response.lower() == 'y':
            demo_with_webcam()
    else:
        print("Algorithm test failed - skipping webcam demo")

if __name__ == "__main__":
    main()