"""
Non-interactive script to run UBFC evaluation
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.evaluate_ubfc import UBFCEvaluator
import time
import pandas as pd

def main():
    # Dataset configuration
    dataset_root = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\data"

    # Create evaluator
    evaluator = UBFCEvaluator(dataset_root)

    print("Running UBFC-rPPG evaluation on all available subjects...")
    print("This may take a while...")

    # Run full evaluation
    results = evaluator.evaluate_dataset()

    if results and results.get('results'):
        # Generate report
        evaluator.generate_report(results, save_path="ubfc_evaluation_report.txt")

        # Save detailed results
        timestamp = int(time.time())
        results_file = f"ubfc_results_{timestamp}.csv"
        pd.DataFrame(results['results']).to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

        # Try to create plots (may fail if display not available)
        try:
            evaluator.plot_evaluation_results(results, save_path="ubfc_evaluation_plots.png")
        except Exception as e:
            print(f"Could not generate plots: {e}")

    else:
        print("No evaluation results generated")

if __name__ == "__main__":
    main()
