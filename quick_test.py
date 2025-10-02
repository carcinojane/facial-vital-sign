"""
Quick test on subject1 only
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.evaluate_ubfc import UBFCEvaluator

def main():
    dataset_root = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\data"
    evaluator = UBFCEvaluator(dataset_root)

    print("Testing on subject1 only...")
    result = evaluator.evaluate_subject("subject1")

    if result and result['metrics']:
        print("\n" + "="*60)
        print("QUICK TEST RESULTS")
        print("="*60)
        metrics = result['metrics']
        print(f"Subject: {metrics['subject']}")
        print(f"MAE: {metrics['mae']:.2f} BPM")
        print(f"RMSE: {metrics['rmse']:.2f} BPM")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Correlation: {metrics['correlation']:.3f}")
        print(f"Within 5 BPM: {metrics['within_5bpm']:.1f}%")
        print(f"Within 10 BPM: {metrics['within_10bpm']:.1f}%")
        print(f"Processing time: {metrics['processing_time']:.1f}s")
        print("="*60)
    else:
        print("Evaluation failed")

if __name__ == "__main__":
    main()
