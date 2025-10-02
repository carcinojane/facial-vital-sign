"""
Evaluate all subjects that have video files
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.evaluate_ubfc import UBFCEvaluator
import pandas as pd
import time

def main():
    dataset_root = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\data"
    evaluator = UBFCEvaluator(dataset_root)

    # List of subjects with video files
    subjects_with_videos = ['subject1', 'subject3', 'subject4', 'subject5',
                           'subject12', 'subject14', 'subject23', 'subject25',
                           'subject26', 'subject27', 'subject48']

    print(f"Running evaluation on {len(subjects_with_videos)} subjects with video files...")
    results = evaluator.evaluate_dataset(subject_list=subjects_with_videos)

    if results and results.get('results'):
        # Generate report
        evaluator.generate_report(results, save_path="ubfc_evaluation_report_final.txt")

        # Save detailed results
        timestamp = int(time.time())
        results_file = f"ubfc_results_final_{timestamp}.csv"
        pd.DataFrame(results['results']).to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

    else:
        print("No evaluation results generated")

if __name__ == "__main__":
    main()
