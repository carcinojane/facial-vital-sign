"""
Run UBFC dataset augmentation to simulate mobile phone quality.

Usage:
    python run_augment_ubfc.py --preview  # Preview on one video
    python run_augment_ubfc.py --sample   # Process one subject
    python run_augment_ubfc.py            # Process full dataset
"""
import subprocess
import os
import sys
import argparse

# Paths
conda_exe = r"C:\Users\janej\anaconda3\Scripts\conda.exe"
script_path = r"C:\Users\janej\OneDrive - National University of Singapore\Capstone Project\rppg-vscode-starter\scripts\augment_ubfc_mobile.py"

def main():
    parser = argparse.ArgumentParser(description='Run UBFC augmentation')
    parser.add_argument('--preview', action='store_true', help='Preview on single video')
    parser.add_argument('--sample', action='store_true', help='Process only first subject')
    parser.add_argument('--preset', type=str, default='moderate',
                       choices=['light', 'moderate', 'heavy'],
                       help='Augmentation intensity')

    args = parser.parse_args()

    # Build command
    cmd = [
        conda_exe, "run",
        "-n", "rppg",
        "python", script_path
    ]

    if args.preview:
        cmd.extend(["--preview", "data/UBFC/subject1/vid.avi"])
    elif args.sample:
        cmd.append("--sample")

    cmd.extend(["--preset", args.preset])

    print("="*70)
    print("UBFC Dataset Augmentation for Mobile Phone Quality")
    print("="*70)
    print(f"Preset: {args.preset}")

    if args.preview:
        print("Mode: Preview (single video)")
    elif args.sample:
        print("Mode: Sample (first subject only)")
    else:
        print("Mode: Full dataset (37 subjects)")

    print("="*70)
    print()

    # Run command
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )

        print(f"\n{'='*70}")
        if result.returncode == 0:
            print("Augmentation completed!")
            if not args.preview:
                print("\nNext steps:")
                print("  1. Verify: Check data/UBFC_MOBILE/")
                print("  2. Update physnet_mobile_finetune.yaml DATA_PATH to data/UBFC_MOBILE")
                print("  3. Run: python run_mobile_finetune.py")
        else:
            print(f"Augmentation failed with exit code: {result.returncode}")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nAugmentation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running augmentation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
