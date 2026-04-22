import os
from segmentation import process_video_for_segmentation

# 1. SETUP PATHS
VIDEO_INPUT = "data/raw_videos/test_sample.mp4" # Put a 5-10 second video here!
CHECKPOINT_DIR = "models/checkpoints"

# 2. DEFINE PROGRESS PRINTER
def print_progress(current, total, message):
    percent = (current / total) * 100
    print(f"[{percent:.1f}%] {message}")

# 3. RUN THE PIPELINE
print("🚀 Starting MoodPlay Perception Pipeline Test...")

try:
    results = process_video_for_segmentation(
        video_path=VIDEO_INPUT,
        progress_callback=print_progress,
        cotracker_checkpoint=f"{CHECKPOINT_DIR}/cotracker/cotracker3.pth",
        sam2_model_cfg="configs/perception/sam2_hiera_l.yaml",
        sam2_checkpoint=f"{CHECKPOINT_DIR}/sam2/sam2_hiera_large.pt",
        device="cuda"
    )

    print("\n✅ PIPELINE SUCCESS!")
    print(f"Total Frames Processed: {results['total_frames']}")
    print(f"Instances Found: {len(results['instances'])}")
    print(f"Check your masks at: {results['mask_root']}")

except Exception as e:
    print(f"\n❌ PIPELINE FAILED: {e}")