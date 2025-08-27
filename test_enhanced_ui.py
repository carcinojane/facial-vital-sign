"""
Test script for enhanced vital signs UI
Demonstrates the new comprehensive monitoring capabilities
"""
import cv2
import numpy as np
import time

def test_ui_components():
    """Test the enhanced UI components"""
    print("Testing Enhanced Vital Signs UI Components...")
    print("=" * 50)
    
    # Test 1: Import advanced vitals processor
    try:
        from scripts.advanced_vitals_ui import AdvancedVitalsProcessor, AdvancedVitalsApp
        print("OK Advanced vitals processor imported successfully")
    except Exception as e:
        print(f"✗ Error importing advanced vitals: {e}")
        return False
    
    # Test 2: Test processor initialization
    try:
        processor = AdvancedVitalsProcessor()
        print("✓ Advanced processor initialized")
        
        # Test vital signs calculation with synthetic data
        synthetic_rgb = []
        for i in range(300):  # 10 seconds at 30 FPS
            # Simulate realistic RGB values with physiological variations
            hr_freq = 1.2  # 72 BPM
            rr_freq = 0.25  # 15 breaths per minute
            
            hr_signal = 3 * np.sin(2 * np.pi * hr_freq * i / 30)
            rr_signal = 1.5 * np.sin(2 * np.pi * rr_freq * i / 30)
            
            rgb = [
                120 + hr_signal + np.random.normal(0, 1),  # Red with HR variation
                95 + hr_signal * 0.7 + rr_signal + np.random.normal(0, 0.8),  # Green with both
                75 + hr_signal * 0.3 + np.random.normal(0, 0.5)  # Blue
            ]
            synthetic_rgb.append(rgb)
        
        # Test different vital sign estimations
        hr = processor.estimate_heart_rate_enhanced([x[0] for x in synthetic_rgb])
        rr = processor.estimate_respiratory_rate(synthetic_rgb)
        spo2 = processor.estimate_spo2(synthetic_rgb[-90:])  # Last 3 seconds
        hrv = processor.calculate_hrv([70, 72, 68, 75, 69, 71, 73])  # Sample HR values
        
        print(f"✓ Synthetic vital signs calculated:")
        print(f"  Heart Rate: {hr:.1f} BPM (target: ~72)")
        print(f"  Respiratory Rate: {rr:.1f} BrPM (target: ~15)")
        print(f"  SpO2: {spo2:.1f}% (target: ~95-99)")
        print(f"  HRV: {hrv:.1f} ms")
        
    except Exception as e:
        print(f"✗ Error testing processor: {e}")
        return False
    
    # Test 3: Dashboard imports
    try:
        from scripts.vitals_dashboard import VitalsDashboard
        print("✓ Dashboard imported successfully")
        
        dashboard = VitalsDashboard()
        print("✓ Dashboard initialized")
        
    except Exception as e:
        print(f"✗ Error importing dashboard: {e}")
        return False
    
    # Test 4: Camera availability
    print("\nTesting camera for UI demonstration...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Camera available for live demonstration")
        
        # Test reading a frame
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera frame captured: {frame.shape}")
        else:
            print("✗ Cannot read from camera")
            
        cap.release()
    else:
        print("✗ Camera not available")
    
    print("\n" + "=" * 50)
    print("Enhanced UI Components Test Complete")
    return True

def demo_selection_menu():
    """Show demo selection menu"""
    print("\nENHANCED VITAL SIGNS UI DEMOS\n")
    print("Choose your experience:")
    print("")
    print("1. BASIC ADVANCED UI")
    print("   - Comprehensive vital signs display")
    print("   - Heart Rate, Respiratory Rate, SpO2")
    print("   - Stress level monitoring")
    print("   - Enhanced face detection")
    print("")
    print("2. FULL DASHBOARD")
    print("   - All basic features PLUS:")
    print("   - Real-time graphs")
    print("   - Session statistics")
    print("   - Data export capabilities")
    print("   - Multiple visualization modes")
    print("")
    print("3. COMPONENT TESTS")
    print("   - Test individual components")
    print("   - Synthetic data validation")
    print("   - Performance benchmarks")
    print("")
    print("4. FEATURE COMPARISON")
    print("   - Compare basic vs advanced UI")
    print("   - Feature matrix")
    print("")
    
    while True:
        try:
            choice = input("Enter your choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            return None

def run_basic_advanced_ui():
    """Run the basic advanced UI"""
    print("\nStarting Basic Advanced UI...")
    print("Features enabled:")
    print("- Heart Rate monitoring")
    print("- Respiratory Rate estimation")
    print("- SpO2 (oxygen saturation) estimation")
    print("- Stress level calculation")
    print("- Signal quality assessment")
    print("- Enhanced face detection with multiple ROI")
    print("")
    print("Controls:")
    print("- Q: Quit")
    print("- S: Save vital signs data")
    print("- R: Reset all data")
    print("- SPACE: Take screenshot")
    print("")
    
    try:
        from scripts.advanced_vitals_ui import AdvancedVitalsApp
        app = AdvancedVitalsApp()
        app.run()
    except Exception as e:
        print(f"Error running advanced UI: {e}")

def run_full_dashboard():
    """Run the full dashboard"""
    print("\nStarting Full Dashboard...")
    print("Features enabled:")
    print("- All basic advanced UI features")
    print("- Real-time plotting with matplotlib")
    print("- Session duration tracking")
    print("- Comprehensive data export")
    print("- Statistical summaries")
    print("- Multiple vital signs graphs")
    print("")
    print("Controls:")
    print("- Q: Quit")
    print("- S: Save session data + summary")
    print("- R: Reset all data")
    print("- D: Toggle live graphs dashboard")
    print("- P: Pause/Resume monitoring")
    print("- SPACE: Take screenshot")
    print("")
    
    try:
        from scripts.vitals_dashboard import VitalsDashboard
        dashboard = VitalsDashboard()
        dashboard.run()
    except Exception as e:
        print(f"Error running dashboard: {e}")

def show_feature_comparison():
    """Show feature comparison matrix"""
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON MATRIX")
    print("=" * 70)
    
    features = [
        ("Feature", "Basic UI", "Advanced UI", "Full Dashboard"),
        ("-" * 30, "-" * 10, "-" * 12, "-" * 15),
        ("Heart Rate Detection", "✓", "✓", "✓"),
        ("Face Detection", "Basic", "Enhanced", "Enhanced"),
        ("ROI Selection", "Single", "Multiple", "Multiple"),
        ("Respiratory Rate", "✗", "✓", "✓"),
        ("SpO2 Estimation", "✗", "✓", "✓"),
        ("Stress Level", "✗", "✓", "✓"),
        ("Heart Rate Variability", "✗", "✓", "✓"),
        ("Signal Quality", "✗", "✓", "✓"),
        ("Real-time Graphs", "✗", "✗", "✓"),
        ("Session Statistics", "Basic", "Basic", "Comprehensive"),
        ("Data Export", "CSV", "CSV", "CSV + Summary"),
        ("Visual Status", "Text", "Colors", "Colors + Bars"),
        ("Screenshot Capture", "✗", "✓", "✓"),
        ("Pause/Resume", "✗", "✗", "✓"),
        ("Live Dashboard", "✗", "✗", "✓")
    ]
    
    for feature in features:
        print(f"{feature[0]:<30} {feature[1]:<10} {feature[2]:<12} {feature[3]:<15}")
    
    print("\nRECOMMENDATIONS:")
    print("- Basic UI: Quick HR testing")
    print("- Advanced UI: Comprehensive monitoring")  
    print("- Full Dashboard: Research & analysis")

def main():
    """Main test and demo function"""
    print("ENHANCED VITAL SIGNS UI - TEST & DEMO")
    print("=" * 50)
    
    # Run component tests first
    if not test_ui_components():
        print("⚠️  Some components failed. Proceeding with available features...")
    
    while True:
        choice = demo_selection_menu()
        
        if choice is None:
            print("Goodbye!")
            break
        elif choice == 1:
            run_basic_advanced_ui()
        elif choice == 2:
            run_full_dashboard()
        elif choice == 3:
            test_ui_components()
        elif choice == 4:
            show_feature_comparison()
        
        # Ask if user wants to continue
        continue_choice = input("\nTry another demo? (y/n): ").lower().strip()
        if continue_choice != 'y':
            break
    
    print("\nThank you for testing the Enhanced Vital Signs UI!")

if __name__ == "__main__":
    main()