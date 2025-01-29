import os
import cv2
import re
import numpy as np
import torch
import json
import pandas as pd
import subprocess
from sam2.build_sam import build_sam2_video_predictor
from datetime import datetime
import gc
import matplotlib.pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
FRAME_SKIP = 5
BATCH_SIZE = 8
DOWNSCALE_FACTOR = 0.5
USE_FP16 = True


def rename_frames_numeric(frame_dir):
    """Remove prefixes from frame filenames"""
    for f in os.listdir(frame_dir):
        if "_" in f:  # Remove any prefixes before numbers
            new_name = f.split("_")[-1]
            os.rename(os.path.join(frame_dir, f), os.path.join(frame_dir, new_name))


def natural_sort_key(s):
    """Natural sorting for numeric filenames"""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split("(\d+)", s)
    ]


class SquashBallTracker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.predictor = self.initialize_model()
        self.ball_positions = []
        self.current_chunk = 0

    def initialize_model(self):
        sam2_checkpoint = "trained-models/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=self.device, half_precision=True
        )
        return predictor

    def process_video_chunk(self, chunk_path, initial_point=None):
        frame_paths = sorted(
            [
                os.path.join(chunk_path, f)
                for f in os.listdir(chunk_path)
                if f.endswith((".jpg", ".png"))
            ],
            key=lambda x: natural_sort_key(x),
        )

        if not frame_paths:
            return []

        if initial_point is None:
            first_frame = cv2.imread(frame_paths[0])
            initial_point = self.get_initial_point(first_frame)

        batch_frames = []
        chunk_positions = []
        inference_state = self.predictor.init_state(video_path=chunk_path)

        for idx, frame_path in enumerate(frame_paths):
            if idx % FRAME_SKIP != 0:
                continue

            frame = cv2.imread(frame_path)
            small_frame = cv2.resize(
                frame, (0, 0), fx=DOWNSCALE_FACTOR, fy=DOWNSCALE_FACTOR
            )

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=idx,
                obj_id=1,
                points=np.array([initial_point * DOWNSCALE_FACTOR]),
                labels=np.array([1]),
            )

            if out_mask_logits is not None:
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                if np.any(mask):
                    y, x = np.where(mask)
                    x_center = int(np.mean(x) / DOWNSCALE_FACTOR)
                    y_center = int(np.mean(y) / DOWNSCALE_FACTOR)

                    chunk_positions.append(
                        {
                            "chunk": self.current_chunk,
                            "frame": idx,
                            "timestamp": idx / 30,
                            "x": x_center,
                            "y": y_center,
                            "confidence": float(out_mask_logits[0].mean().item()),
                        }
                    )

            # Clear memory every few frames
            if idx % BATCH_SIZE == 0:
                del out_mask_logits
                torch.cuda.empty_cache()

        # Final cleanup
        del inference_state
        torch.cuda.empty_cache()
        gc.collect()

        self.current_chunk += 1
        return chunk_positions

    def get_initial_point(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(frame_rgb)
        plt.title("Click on the squash ball")
        point = plt.ginput(1)[0]
        plt.close()
        return np.array(point, dtype=np.float32)

    def save_results(self, format="csv"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format == "csv":
            df = pd.DataFrame(self.ball_positions)
            df.to_csv(f"ball_positions_{timestamp}.csv", index=False)
        elif format == "json":
            with open(f"ball_positions_{timestamp}.json", "w") as f:
                json.dump(self.ball_positions, f, indent=2)
        print(f"Results saved in {format.upper()} format")


def process_long_video(video_path, chunk_length=120):
    tracker = SquashBallTracker()
    os.makedirs("video_chunks", exist_ok=True)
    safe_video_path = os.path.abspath(video_path).replace("\\", "/")

    try:
        subprocess.run(
            f'ffmpeg -i "{safe_video_path}" -c copy -f segment -segment_time {chunk_length} '
            f'-reset_timestamps 1 "video_chunks/chunk_%03d.mp4"',
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
        return

    chunks = sorted(
        [f for f in os.listdir("video_chunks") if f.endswith(".mp4")],
        key=natural_sort_key,
    )

    initial_point = None
    for chunk in chunks:
        chunk_path = os.path.join("video_chunks", chunk)
        frame_dir = f"frames_{tracker.current_chunk}"
        os.makedirs(frame_dir, exist_ok=True)

        try:
            # Extract frames with numeric-only names
            subprocess.run(
                f'ffmpeg -i "{chunk_path}" "{frame_dir}/%05d.jpg"',
                shell=True,
                check=True,
            )
            # Rename if any prefixes exist
            rename_frames_numeric(frame_dir)

        except subprocess.CalledProcessError as e:
            print(f"Frame extraction failed: {e}")
            continue

        positions = tracker.process_video_chunk(frame_dir, initial_point)
        tracker.ball_positions.extend(positions)

        # Cleanup
        for f in os.listdir(frame_dir):
            os.remove(os.path.join(frame_dir, f))
        os.rmdir(frame_dir)
        os.remove(chunk_path)

        if positions:
            initial_point = np.array([positions[-1]["x"], positions[-1]["y"]])

        tracker.save_results(format="csv")

    os.rmdir("video_chunks")
    tracker.save_results(format="csv")
    return tracker.ball_positions


if __name__ == "__main__":
    video_path = r"farag_elshorbagy_1m_chopped.mp4"
    process_long_video(video_path, chunk_length=120)
