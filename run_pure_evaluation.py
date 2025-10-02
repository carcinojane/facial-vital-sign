"""
Non-interactive script to run PURE evaluation and generate results.txt
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.evaluate_pure import PUREEvaluator
import time
import pandas as pd

def main():
    # Dataset configuration
    dataset_root = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\data"

    # Create evaluator
    evaluator = PUREEvaluator(dataset_root, fps=30)

    print("Running PURE dataset evaluation on all available subjects...")
    print("This may take a while...")

    # Run full evaluation
    results = evaluator.evaluate_dataset()

    if results and results.get('results'):
        # Generate report and save to results.txt
        evaluator.generate_report(results, save_path="results.txt")

        # Save detailed results
        timestamp = int(time.time())
        results_file = f"pure_results_{timestamp}.csv"
        pd.DataFrame(results['results']).to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")

        # Try to create plots (may fail if display not available)
        try:
            evaluator.plot_evaluation_results(results, save_path="pure_evaluation_plots.png")
        except Exception as e:
            print(f"Could not generate plots: {e}")

        print(f"\n{'='*70}")
        print("Evaluation complete! Results saved to results.txt")
        print(f"{'='*70}")

    else:
        print("No evaluation results generated")

if __name__ == "__main__":
    main()
