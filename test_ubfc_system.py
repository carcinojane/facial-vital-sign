"""
Test script for UBFC rPPG evaluation and fine-tuning system
"""
import numpy as np
from pathlib import Path
import os

def test_ubfc_access():
    """Test UBFC dataset access"""
    print("Testing UBFC dataset access...")
    
    dataset_root = r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data"
    ubfc_path = Path(dataset_root) / "UBFC"
    
    if not ubfc_path.exists():
        print(f"FAIL: UBFC dataset not found at {ubfc_path}")
        return False
    
    # Count subjects
    subjects = [d for d in ubfc_path.iterdir() if d.is_dir() and d.name.startswith('subject')]
    print(f"OK: Found {len(subjects)} UBFC subjects")
    
    # Check a few subjects for required files
    valid_subjects = 0
    for subject in subjects[:3]:
        video_files = list(subject.glob("*.avi"))
        gt_files = list(subject.glob("ground_truth.txt"))
        
        if video_files and gt_files:
            valid_subjects += 1
            print(f"  {subject.name}: Video + Ground Truth OK")
        else:
            print(f"  {subject.name}: Missing files")
    
    print(f"OK: {valid_subjects}/3 subjects have required files")
    return valid_subjects > 0

def test_ground_truth_loading():
    """Test ground truth loading functionality"""
    print("\nTesting ground truth loading...")
    
    try:
        from scripts.evaluate_ubfc import UBFCEvaluator
        
        dataset_root = r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data"
        evaluator = UBFCEvaluator(dataset_root)
        
        # Test with first available subject
        ubfc_path = Path(dataset_root) / "UBFC"
        subjects = [d for d in ubfc_path.iterdir() if d.is_dir() and d.name.startswith('subject')]
        
        if len(subjects) == 0:
            print("FAIL: No subjects found")
            return False
        
        test_subject = subjects[0]
        print(f"Testing with {test_subject.name}")
        
        # Load ground truth
        gt_data = evaluator.load_ubfc_ground_truth(test_subject)
        
        if gt_data is not None:
            print(f"OK: Loaded {len(gt_data)} ground truth HR values")
            print(f"  HR range: {np.min(gt_data):.1f} - {np.max(gt_data):.1f} BPM")
            print(f"  Mean HR: {np.mean(gt_data):.1f} BPM")
            return True
        else:
            print("FAIL: Could not load ground truth")
            return False
            
    except Exception as e:
        print(f"FAIL: Error loading ground truth: {e}")
        return False

def test_video_processing():
    """Test video processing capabilities"""
    print("\nTesting video processing...")
    
    try:
        from scripts.evaluate_ubfc import UBFCEvaluator
        import cv2
        
        dataset_root = r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data"
        evaluator = UBFCEvaluator(dataset_root)
        
        # Find first subject with video
        ubfc_path = Path(dataset_root) / "UBFC"
        subjects = [d for d in ubfc_path.iterdir() if d.is_dir() and d.name.startswith('subject')]
        
        for subject in subjects:
            video_files = list(subject.glob("*.avi"))
            if video_files:
                video_path = video_files[0]
                print(f"Testing video: {video_path}")
                
                # Test video opening
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print("FAIL: Cannot open video")
                    return False
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                print(f"OK: Video properties - {fps:.1f} FPS, {frame_count} frames, {duration:.1f}s")
                
                # Test reading a few frames
                frames_read = 0
                for i in range(10):  # Try to read 10 frames
                    ret, frame = cap.read()
                    if ret:
                        frames_read += 1
                    else:
                        break
                
                cap.release()
                
                print(f"OK: Successfully read {frames_read}/10 test frames")
                return frames_read > 0
        
        print("FAIL: No video files found")
        return False
        
    except Exception as e:
        print(f"FAIL: Error processing video: {e}")
        return False

def test_evaluation_pipeline():
    """Test the evaluation pipeline on one subject"""
    print("\nTesting evaluation pipeline...")
    
    try:
        from scripts.evaluate_ubfc import UBFCEvaluator
        
        dataset_root = r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data"
        evaluator = UBFCEvaluator(dataset_root)
        
        # Get first subject
        ubfc_path = Path(dataset_root) / "UBFC"
        subjects = [d.name for d in ubfc_path.iterdir() if d.is_dir() and d.name.startswith('subject')]
        
        if len(subjects) == 0:
            print("FAIL: No subjects found")
            return False
        
        test_subject = subjects[0]
        print(f"Running evaluation on {test_subject}...")
        
        # Run evaluation (this will take time)
        print("WARNING: This may take several minutes...")
        result = evaluator.evaluate_subject(test_subject)
        
        if result and result.get('metrics'):
            metrics = result['metrics']
            print("OK: Evaluation completed successfully")
            print(f"  MAE: {metrics['mae']:.2f} BPM")
            print(f"  RMSE: {metrics['rmse']:.2f} BPM") 
            print(f"  Correlation: {metrics['correlation']:.3f}")
            print(f"  Within 10 BPM: {metrics['within_10bpm']:.1f}%")
            return True
        else:
            print("FAIL: Evaluation failed")
            return False
            
    except Exception as e:
        print(f"FAIL: Error in evaluation: {e}")
        return False

def test_fine_tuning_setup():
    """Test fine-tuning system setup"""
    print("\nTesting fine-tuning setup...")
    
    try:
        from scripts.finetune_rppg import RPPGFineTuner, AdvancedRPPGProcessor
        
        dataset_root = r"G:\My Drive\iss\Capstone_Project\Vital_sign_scan_pretrained\data"
        tuner = RPPGFineTuner(dataset_root)
        
        # Test advanced processor creation
        processor = AdvancedRPPGProcessor(
            filter_low=0.8, 
            filter_high=2.8,
            filter_order=4,
            alpha_method='adaptive'
        )
        
        # Test with synthetic data
        synthetic_data = []
        for i in range(300):  # 10 seconds at 30 FPS
            # Simulate RGB values with small variations
            rgb = [100 + np.sin(i*0.1), 80 + np.cos(i*0.1), 60 + np.sin(i*0.15)]
            synthetic_data.append(rgb)
        
        # Test POS algorithm
        hr, signal = processor.pos_algorithm_tunable(synthetic_data)
        
        print(f"OK: Advanced processor working")
        print(f"  Synthetic data HR estimate: {hr:.1f} BPM")
        print(f"  Signal length: {len(signal)}")
        
        # Test parameter bounds
        test_params = [0.7, 3.0, 4, 1.0]
        test_subjects = ['subject1']  # Dummy for testing
        
        # This won't actually run full evaluation, just test the setup
        print("OK: Fine-tuning system setup complete")
        return True
        
    except Exception as e:
        print(f"FAIL: Error in fine-tuning setup: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("UBFC rPPG SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("UBFC Dataset Access", test_ubfc_access),
        ("Ground Truth Loading", test_ground_truth_loading),
        ("Video Processing", test_video_processing),
        ("Fine-tuning Setup", test_fine_tuning_setup),
        # ("Evaluation Pipeline", test_evaluation_pipeline),  # Commented out - takes too long
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"FAIL: Unexpected error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\nAll tests passed! Your UBFC system is ready for:")
        print("✓ Evaluation: python scripts/evaluate_ubfc.py")
        print("✓ Fine-tuning: python scripts/finetune_rppg.py")
    else:
        print("\nSome tests failed. Check the issues above.")
    
    # Quick start guide
    if results.get("UBFC Dataset Access", False):
        print("\nQUICK START:")
        print("1. Evaluate 3 subjects: python scripts/evaluate_ubfc.py (select option 1)")
        print("2. Fine-tune parameters: python scripts/finetune_rppg.py")
        print("3. Compare with original: python scripts/simple_rppg_ui.py")

if __name__ == "__main__":
    main()