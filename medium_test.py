"""
Medium test on 10 subjects
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

    print("Running evaluation on 10 subjects...")
    results = evaluator.evaluate_dataset(max_subjects=10)

    if results and results.get('results'):
        # Generate report
        evaluator.generate_report(results, save_path="ubfc_evaluation_report.txt")

        # Save detailed results
        timestamp = int(time.time())
        results_file = f"ubfc_results_{timestamp}.csv"
        pd.DataFrame(results['results']).to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

    else:
        print("No evaluation results generated")

if __name__ == "__main__":
    main()
