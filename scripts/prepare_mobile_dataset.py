"""
Script to prepare mobile phone video dataset for PhysNet fine-tuning.

This script helps organize mobile phone videos in the expected format for the rPPG-Toolbox.
Supports UBFC-rPPG format or can be adapted for custom formats.

Expected mobile video dataset structure:
data/MOBILE/
├── subject1/
│   ├── vid.avi (or .mp4)
│   └── ground_truth.txt (heart rate ground truth)
├── subject2/
│   ├── vid.avi
│   └── ground_truth.txt
└── ...

Ground truth format (ground_truth.txt):
- Each line: timestamp, heart_rate
- OR: Single file with HR values at fixed sampling rate
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import json
import shutil

def validate_video(video_path):
    """Validate that video can be read and has reasonable properties."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "Cannot open video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps < 15 or fps > 60:
        return False, f"Unusual FPS: {fps}"
    if frame_count < 90:  # Less than 3 seconds at 30fps
        return False, f"Video too short: {frame_count} frames"
    if width < 240 or height < 240:
        return False, f"Resolution too low: {width}x{height}"

    return True, f"Valid: {frame_count} frames, {fps:.1f} fps, {width}x{height}"

def validate_ground_truth(gt_path):
    """Validate ground truth file."""
    if not os.path.exists(gt_path):
        return False, "Ground truth file not found"

    try:
        # Try to read as space/comma separated values
        with open(gt_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            return False, "Empty ground truth file"

        # Check if it's a single line (continuous HR value)
        if len(lines) == 1:
            values = lines[0].strip().split()
            if len(values) > 10:  # Looks like continuous data
                return True, f"Continuous format: {len(values)} values"

        # Check if it's timestamp, HR pairs
        first_line = lines[0].strip().split()
        if len(first_line) >= 2:
            try:
                float(first_line[0])  # timestamp
                float(first_line[1])  # HR value
                return True, f"Timestamp format: {len(lines)} entries"
            except ValueError:
                pass

        return False, "Unknown ground truth format"

    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def organize_dataset(input_dir, output_dir, video_ext='.avi'):
    """
    Organize videos and ground truth files into UBFC-rPPG compatible format.

    Args:
        input_dir: Directory containing raw mobile videos
        output_dir: Output directory (data/MOBILE)
        video_ext: Video file extension to look for
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = list(input_path.rglob(f'*{video_ext}'))
    if video_ext != '.mp4':
        video_files += list(input_path.rglob('*.mp4'))

    print(f"Found {len(video_files)} video files")

    valid_count = 0
    invalid_count = 0

    for i, video_file in enumerate(video_files, 1):
        # Determine subject directory name
        subject_name = f"subject{i}"
        subject_dir = output_path / subject_name
        subject_dir.mkdir(exist_ok=True)

        # Validate video
        is_valid, msg = validate_video(video_file)
        print(f"\n{i}. {video_file.name}")
        print(f"   {msg}")

        if not is_valid:
            invalid_count += 1
            print(f"   SKIPPED")
            continue

        # Copy video
        output_video = subject_dir / 'vid.avi'
        shutil.copy2(video_file, output_video)
        print(f"   Copied to: {output_video}")

        # Look for corresponding ground truth file
        # Check common naming patterns
        gt_candidates = [
            video_file.with_suffix('.txt'),
            video_file.parent / 'ground_truth.txt',
            video_file.parent / 'hr.txt',
            video_file.parent / f'{video_file.stem}_gt.txt',
        ]

        gt_found = False
        for gt_file in gt_candidates:
            if gt_file.exists():
                is_valid_gt, gt_msg = validate_ground_truth(gt_file)
                print(f"   GT: {gt_msg}")

                if is_valid_gt:
                    # Copy ground truth
                    output_gt = subject_dir / 'ground_truth.txt'
                    shutil.copy2(gt_file, output_gt)
                    print(f"   GT copied to: {output_gt}")
                    gt_found = True
                    valid_count += 1
                    break

        if not gt_found:
            print(f"   WARNING: No valid ground truth found")
            print(f"   Creating placeholder GT file (NEEDS MANUAL FILLING)")
            # Create placeholder
            with open(subject_dir / 'ground_truth.txt', 'w') as f:
                f.write("# PLACEHOLDER - Add ground truth HR values here\n")
                f.write("# Format: one value per line (for continuous) or timestamp,hr pairs\n")
            invalid_count += 1

    print(f"\n{'='*60}")
    print(f"Dataset organization complete!")
    print(f"Valid subjects: {valid_count}")
    print(f"Invalid/incomplete subjects: {invalid_count}")
    print(f"Output directory: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Review subjects with placeholder GT files and add ground truth")
    print(f"2. Update physnet_mobile_finetune.yaml DATA_PATH to: {output_path}")
    print(f"3. Run fine-tuning: python run_mobile_finetune.py")

def create_sample_structure(output_dir):
    """Create sample dataset structure for reference."""
    output_path = Path(output_dir)
    sample_dir = output_path / "subject_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create README
    readme_content = """# Mobile Phone Video Dataset Structure

This directory shows the expected structure for mobile phone videos.

Each subject should have:
1. vid.avi (or .mp4) - The video file
2. ground_truth.txt - Heart rate ground truth

## Ground Truth Format Options:

### Option 1: Continuous HR values (one per line)
```
75.2
76.1
74.8
...
```

### Option 2: Timestamp, HR pairs (space or comma separated)
```
0.0 75.2
0.5 76.1
1.0 74.8
...
```

### Option 3: Single line with space-separated values
```
75.2 76.1 74.8 75.5 ...
```

## Video Requirements:
- Format: .avi or .mp4
- FPS: 15-60 fps (30 fps recommended)
- Resolution: At least 240x240 (640x480 or higher recommended)
- Duration: At least 30 seconds (60+ seconds recommended)
- Content: Face clearly visible, good lighting
"""

    with open(sample_dir / 'README.txt', 'w') as f:
        f.write(readme_content)

    # Create sample ground truth
    with open(sample_dir / 'ground_truth.txt', 'w') as f:
        f.write("# Sample ground truth file\n")
        f.write("# Replace with actual heart rate measurements\n")
        for t in np.arange(0, 60, 0.5):  # 60 seconds, 2 Hz sampling
            hr = 75 + 5 * np.sin(2 * np.pi * 0.1 * t)  # Simulated varying HR
            f.write(f"{t:.1f} {hr:.1f}\n")

    print(f"\nSample structure created in: {sample_dir}")
    print(f"Review README.txt for format details")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare mobile phone video dataset for PhysNet fine-tuning')
    parser.add_argument('--input', type=str, help='Input directory containing mobile videos')
    parser.add_argument('--output', type=str,
                       default='data/MOBILE',
                       help='Output directory (default: data/MOBILE)')
    parser.add_argument('--ext', type=str, default='.avi',
                       help='Video file extension (default: .avi)')
    parser.add_argument('--sample', action='store_true',
                       help='Create sample dataset structure for reference')

    args = parser.parse_args()

    if args.sample:
        create_sample_structure(args.output)
    elif args.input:
        organize_dataset(args.input, args.output, args.ext)
    else:
        print("Error: Either --input or --sample must be specified")
        print("\nUsage examples:")
        print("  Create sample structure:")
        print("    python scripts/prepare_mobile_dataset.py --sample")
        print("\n  Organize existing videos:")
        print("    python scripts/prepare_mobile_dataset.py --input raw_videos/")
        print("    python scripts/prepare_mobile_dataset.py --input raw_videos/ --ext .mp4")
        parser.print_help()
