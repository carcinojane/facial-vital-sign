"""
Evaluate fine-tuned PhysNet model on mobile phone videos.

This script loads the fine-tuned model and evaluates it on mobile test videos,
providing detailed metrics and comparisons with the baseline model.
"""
import os
import sys
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path

# Add rppg_toolbox to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rppg_toolbox'))

from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX

def load_model(model_path, device='cpu'):
    """Load PhysNet model from checkpoint."""
    model = PhysNet_padding_Encoder_Decoder_MAX(frames=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def compare_models(baseline_path, finetuned_path, test_videos_dir):
    """
    Compare baseline and fine-tuned models on test videos.

    Args:
        baseline_path: Path to baseline PURE/UBFC model
        finetuned_path: Path to fine-tuned mobile model
        test_videos_dir: Directory containing test videos
    """
    print("="*70)
    print("PhysNet Model Comparison: Baseline vs Fine-tuned")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load models
    print("\nLoading models...")
    baseline_model = load_model(baseline_path, device)
    finetuned_model = load_model(finetuned_path, device)
    print("✓ Models loaded")

    # TODO: Implement full evaluation pipeline
    # This would require implementing the preprocessing and inference logic
    # For now, just show model info

    print(f"\nBaseline model: {baseline_path}")
    print(f"Fine-tuned model: {finetuned_path}")
    print(f"\nTest videos directory: {test_videos_dir}")

    print("\n" + "="*70)
    print("To run full evaluation, use the rPPG-Toolbox main script:")
    print("  python rppg_toolbox/main.py --config_file physnet_mobile_eval.yaml")
    print("="*70)

def quick_test_single_video(model_path, video_path):
    """
    Quick test on a single video file.

    Args:
        model_path: Path to model checkpoint
        video_path: Path to test video
    """
    print("="*70)
    print("Quick Test on Single Video")
    print("="*70)

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")

    # Load model
    model = load_model(model_path, device)
    print("\n✓ Model loaded successfully")

    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    print(f"\nVideo info:")
    print(f"  FPS: {fps:.1f}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {duration:.1f} seconds")

    print("\n" + "="*70)
    print("Note: Full preprocessing and inference requires using the toolbox:")
    print("  1. Add video to data/MOBILE/test_subject/vid.avi")
    print("  2. Run: python rppg_toolbox/main.py --config_file physnet_mobile_eval.yaml")
    print("="*70)

def create_eval_config(model_path, output_path='physnet_mobile_eval.yaml'):
    """Create evaluation config file for the fine-tuned model."""
    config_content = f"""BASE: ['']
TOOLBOX_MODE: "only_test"
DEVICE: cpu  # Change to cuda if available
NUM_OF_GPU_TRAIN: 1

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR']
  USE_LAST_EPOCH: True
  DATA:
    FS: 30
    DATASET: UBFC-rPPG  # Or CUSTOM if you have custom loader
    DO_PREPROCESS: True
    DATA_FORMAT: NCDHW
    DATA_PATH: "C:/Users/janej/OneDrive - National University of Singapore/Capstone Project/rppg-vscode-starter/data/MOBILE"
    CACHED_PATH: "C:/rppg_cache_mobile_test"
    BEGIN: 0.85  # Test set only
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['DiffNormalized']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 128
      CROP_FACE:
        DO_CROP_FACE: True
        BACKEND: 'HC'
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 32
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

MODEL:
  DROP_RATE: 0.2
  NAME: Physnet
  PHYSNET:
    FRAME_NUM: 128

LOG:
  PATH: runs/mobile_eval

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: FFT
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: "{model_path}"
"""

    with open(output_path, 'w') as f:
        f.write(config_content)

    print(f"Created evaluation config: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned PhysNet model on mobile videos')
    parser.add_argument('--model', type=str,
                       help='Path to fine-tuned model checkpoint',
                       default='runs/mobile_finetune/physnet_mobile_finetuned_best.pth')
    parser.add_argument('--baseline', type=str,
                       help='Path to baseline model for comparison',
                       default='rppg_toolbox/final_model_release/PURE_PhysNet_DiffNormalized.pth')
    parser.add_argument('--test-dir', type=str,
                       help='Directory containing test videos',
                       default='data/MOBILE')
    parser.add_argument('--video', type=str,
                       help='Single video file for quick test')
    parser.add_argument('--create-config', action='store_true',
                       help='Create evaluation config file')

    args = parser.parse_args()

    if args.create_config:
        # Create eval config
        if not os.path.exists(args.model):
            print(f"Warning: Model path does not exist: {args.model}")
            print("Creating config anyway with this path...")

        config_path = create_eval_config(args.model)
        print(f"\nTo run evaluation:")
        print(f"  python rppg_toolbox/main.py --config_file {config_path}")

    elif args.video:
        # Quick test on single video
        quick_test_single_video(args.model, args.video)

    else:
        # Compare models
        compare_models(args.baseline, args.model, args.test_dir)

    print("\nUsage examples:")
    print("  Create evaluation config:")
    print("    python scripts/evaluate_mobile_model.py --create-config")
    print("\n  Quick test on single video:")
    print("    python scripts/evaluate_mobile_model.py --video data/MOBILE/subject1/vid.avi")
    print("\n  Compare models:")
    print("    python scripts/evaluate_mobile_model.py --model runs/mobile_finetune/best.pth")
