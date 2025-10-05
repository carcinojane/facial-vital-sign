"""
Augment UBFC-rPPG dataset to simulate mobile phone camera quality.

This script applies realistic degradations to laboratory-quality UBFC videos
to simulate mobile phone recording conditions:
- Compression artifacts (H.264/H.265 encoding)
- Lower resolution
- Motion blur (camera shake)
- Lighting variations
- Noise (sensor noise, low-light conditions)
- Color balance shifts

Usage:
    python scripts/augment_ubfc_mobile.py --input data/UBFC --output data/UBFC_MOBILE
    python scripts/augment_ubfc_mobile.py --sample  # Process single subject for testing
"""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

class MobilePhoneAugmentor:
    """Applies realistic mobile phone camera degradations to videos."""

    def __init__(self, preset='moderate'):
        """
        Initialize augmentor with degradation preset.

        Args:
            preset: 'light', 'moderate', or 'heavy' degradation level
        """
        self.preset = preset
        self.presets = {
            'light': {
                'compression_quality': 85,
                'resolution_scale': 0.8,
                'motion_blur_prob': 0.2,
                'motion_blur_size': 5,
                'noise_sigma': 3,
                'brightness_range': (-15, 15),
                'apply_compression': True,
            },
            'moderate': {
                'compression_quality': 70,
                'resolution_scale': 0.6,
                'motion_blur_prob': 0.3,
                'motion_blur_size': 7,
                'noise_sigma': 5,
                'brightness_range': (-25, 25),
                'apply_compression': True,
            },
            'heavy': {
                'compression_quality': 50,
                'resolution_scale': 0.5,
                'motion_blur_prob': 0.4,
                'motion_blur_size': 9,
                'noise_sigma': 8,
                'brightness_range': (-35, 35),
                'apply_compression': True,
            }
        }
        self.config = self.presets[preset]

    def add_compression_artifacts(self, frame):
        """Simulate H.264 video compression artifacts."""
        if not self.config['apply_compression']:
            return frame

        # JPEG compression to simulate block artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config['compression_quality']]
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return compressed

    def add_motion_blur(self, frame):
        """Add motion blur to simulate camera shake."""
        if np.random.random() > self.config['motion_blur_prob']:
            return frame

        size = self.config['motion_blur_size']

        # Random direction motion blur
        angle = np.random.uniform(0, 180)
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size

        # Rotate kernel to random angle
        M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (size, size))

        blurred = cv2.filter2D(frame, -1, kernel)
        return blurred

    def add_gaussian_noise(self, frame):
        """Add sensor noise (especially for low-light conditions)."""
        noise = np.random.normal(0, self.config['noise_sigma'], frame.shape)
        noisy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def adjust_brightness(self, frame):
        """Simulate varying lighting conditions."""
        brightness_shift = np.random.randint(*self.config['brightness_range'])
        adjusted = np.clip(frame.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)
        return adjusted

    def adjust_color_balance(self, frame):
        """Simulate different white balance settings."""
        # Random color temperature shift
        temp_shift = np.random.uniform(0.9, 1.1)

        # Apply to blue/red channels
        frame_float = frame.astype(np.float32)
        if np.random.random() > 0.5:
            # Warm (increase red/yellow)
            frame_float[:, :, 2] *= temp_shift  # Red
            frame_float[:, :, 0] *= (2.0 - temp_shift)  # Blue
        else:
            # Cool (increase blue)
            frame_float[:, :, 0] *= temp_shift  # Blue
            frame_float[:, :, 2] *= (2.0 - temp_shift)  # Red

        return np.clip(frame_float, 0, 255).astype(np.uint8)

    def resize_and_upscale(self, frame):
        """Simulate lower resolution capture then upscale."""
        h, w = frame.shape[:2]
        scale = self.config['resolution_scale']

        # Downscale
        small = cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

        # Upscale back (simulates lower quality)
        upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return upscaled

    def augment_frame(self, frame):
        """Apply full augmentation pipeline to a single frame."""
        # Apply transformations in realistic order
        frame = self.resize_and_upscale(frame)
        frame = self.add_motion_blur(frame)
        frame = self.add_gaussian_noise(frame)
        frame = self.adjust_brightness(frame)
        frame = self.adjust_color_balance(frame)
        frame = self.add_compression_artifacts(frame)

        return frame

    def augment_video(self, input_path, output_path):
        """
        Augment entire video file.

        Args:
            input_path: Path to input video
            output_path: Path to save augmented video

        Returns:
            success (bool), message (str)
        """
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return False, "Cannot open video"

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup video writer (use H.264 codec for realistic compression)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'mp4v', 'H264'
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            return False, "Cannot create output video"

        # Process frames
        frames_processed = 0
        with tqdm(total=frame_count, desc=f"Augmenting {input_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Augment frame
                augmented = self.augment_frame(frame)
                out.write(augmented)

                frames_processed += 1
                pbar.update(1)

        cap.release()
        out.release()

        return True, f"Processed {frames_processed} frames"


def augment_ubfc_dataset(input_dir, output_dir, preset='moderate', sample_only=False):
    """
    Augment entire UBFC-rPPG dataset to simulate mobile phone quality.

    Args:
        input_dir: Path to original UBFC dataset (data/UBFC)
        output_dir: Path to save augmented dataset (data/UBFC_MOBILE)
        preset: Degradation level ('light', 'moderate', 'heavy')
        sample_only: Only process first subject for testing
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    print("="*70)
    print("UBFC Dataset Augmentation for Mobile Phone Quality")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Preset: {preset}")
    print("="*70)

    augmentor = MobilePhoneAugmentor(preset=preset)

    # Print augmentation settings
    print("\nAugmentation Settings:")
    for key, value in augmentor.config.items():
        print(f"  {key}: {value}")
    print()

    # Find all subject directories
    subject_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])

    if sample_only:
        subject_dirs = subject_dirs[:1]
        print(f"Sample mode: Processing only {subject_dirs[0].name}\n")

    print(f"Found {len(subject_dirs)} subjects to process\n")

    success_count = 0
    fail_count = 0

    for subject_dir in subject_dirs:
        subject_name = subject_dir.name
        print(f"\nProcessing {subject_name}...")

        # Find video file (vid.avi or vid.mp4)
        video_file = None
        for ext in ['.avi', '.mp4']:
            vid_path = subject_dir / f'vid{ext}'
            if vid_path.exists():
                video_file = vid_path
                break

        if not video_file:
            print(f"  [FAILED] No video file found (vid.avi or vid.mp4)")
            fail_count += 1
            continue

        # Create output subject directory
        output_subject = output_path / subject_name
        output_subject.mkdir(parents=True, exist_ok=True)

        # Augment video
        output_video = output_subject / 'vid.avi'
        success, msg = augmentor.augment_video(video_file, output_video)

        if success:
            print(f"  [OK] Video augmented: {msg}")

            # Copy ground truth file
            gt_file = subject_dir / 'ground_truth.txt'
            if gt_file.exists():
                shutil.copy2(gt_file, output_subject / 'ground_truth.txt')
                print(f"  [OK] Ground truth copied")
            else:
                print(f"  [WARNING] No ground truth file found")

            success_count += 1
        else:
            print(f"  [FAILED] Failed: {msg}")
            fail_count += 1

    print("\n" + "="*70)
    print("Augmentation Complete!")
    print("="*70)
    print(f"Successfully processed: {success_count} subjects")
    print(f"Failed: {fail_count} subjects")
    print(f"\nAugmented dataset location: {output_path}")
    print("\nNext steps:")
    print("  1. Verify augmented videos: ls", str(output_path))
    print("  2. Update physnet_mobile_finetune.yaml DATA_PATH to:", str(output_path))
    print("  3. Run fine-tuning: python run_mobile_finetune.py")
    print("="*70)


def preview_augmentation(input_video, output_dir, preset='moderate', num_frames=10):
    """
    Preview augmentation on sample frames from a video.

    Args:
        input_video: Path to input video
        output_dir: Directory to save preview frames
        preset: Augmentation preset
        num_frames: Number of frames to preview
    """
    input_path = Path(input_video)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating augmentation preview from {input_path.name}...")

    augmentor = MobilePhoneAugmentor(preset=preset)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample frames evenly throughout video
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Save original
        cv2.imwrite(str(output_path / f'frame_{i:02d}_original.jpg'), frame)

        # Augment and save
        augmented = augmentor.augment_frame(frame)
        cv2.imwrite(str(output_path / f'frame_{i:02d}_augmented.jpg'), augmented)

    cap.release()

    print(f"[OK] Preview frames saved to: {output_path}")
    print(f"  {num_frames} original + {num_frames} augmented frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Augment UBFC dataset to simulate mobile phone camera quality'
    )

    parser.add_argument('--input', type=str,
                       default='data/UBFC',
                       help='Input UBFC dataset directory (default: data/UBFC)')
    parser.add_argument('--output', type=str,
                       default='data/UBFC_MOBILE',
                       help='Output directory for augmented dataset (default: data/UBFC_MOBILE)')
    parser.add_argument('--preset', type=str,
                       choices=['light', 'moderate', 'heavy'],
                       default='moderate',
                       help='Augmentation intensity (default: moderate)')
    parser.add_argument('--sample', action='store_true',
                       help='Process only first subject for testing')
    parser.add_argument('--preview', type=str,
                       help='Create preview of augmentation from single video file')
    parser.add_argument('--preview-output', type=str,
                       default='augmentation_preview',
                       help='Directory for preview frames (default: augmentation_preview)')

    args = parser.parse_args()

    if args.preview:
        # Preview mode
        preview_augmentation(args.preview, args.preview_output, args.preset)
    else:
        # Full augmentation
        augment_ubfc_dataset(args.input, args.output, args.preset, args.sample)

    print("\nUsage examples:")
    print("\n  Preview augmentation on single video:")
    print("    python scripts/augment_ubfc_mobile.py --preview data/UBFC/subject1/vid.avi")
    print("\n  Process single subject (test):")
    print("    python scripts/augment_ubfc_mobile.py --sample")
    print("\n  Process full dataset:")
    print("    python scripts/augment_ubfc_mobile.py")
    print("\n  Heavy degradation:")
    print("    python scripts/augment_ubfc_mobile.py --preset heavy")
