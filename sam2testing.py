import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import natsort
import csv
from datetime import datetime
import json
import shutil


def initialize_sam2():
    """Initialize the SAM2 model with optimizations."""
    sam2_checkpoint = "trained-models/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    return predictor


def extract_frames_efficiently(video_path, temp_folder="temp_frames"):
    """Extract frames more efficiently using threading."""
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    def save_frame(args):
        frame, index = args
        if frame is not None:
            frame_path = os.path.join(temp_folder, f"{index + 1}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return frame_path
        return None

    frame_paths = []
    frames_to_save = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_to_save.append((frame, len(frame_paths)))
        frame_paths.append(None)

        if len(frames_to_save) >= 32:
            with ThreadPoolExecutor(max_workers=4) as executor:
                saved_paths = list(executor.map(save_frame, frames_to_save))
                for i, path in enumerate(saved_paths):
                    frame_paths[len(frame_paths) - len(frames_to_save) + i] = path
            frames_to_save = []

    if frames_to_save:
        with ThreadPoolExecutor(max_workers=4) as executor:
            saved_paths = list(executor.map(save_frame, frames_to_save))
            for i, path in enumerate(saved_paths):
                frame_paths[len(frame_paths) - len(frames_to_save) + i] = path

    cap.release()
    return [p for p in frame_paths if p is not None], fps


def create_output_directory(input_path):
    """Create an organized output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = f"tracking_results_{video_name}_{timestamp}"

    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)

    return output_dir


def track_squash_ball(input_path):
    try:
        # Create output directory
        output_dir = create_output_directory(input_path)

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log processing parameters
        processing_info = {
            "input_file": input_path,
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cuda_available": torch.cuda.is_available(),
        }

        is_video = input_path.lower().endswith((".mp4", ".avi", ".mov"))

        if is_video:
            print("Processing video file...")
            frame_paths, fps = extract_frames_efficiently(input_path)
            folder_path = os.path.dirname(frame_paths[0])
            processing_info["fps"] = fps
        else:
            print("Processing folder of frames...")
            folder_path = input_path
            frame_paths = natsort.natsorted(
                [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.endswith((".jpg", ".png"))
                ]
            )
            fps = 30.0
            processing_info["fps"] = fps

        predictor = initialize_sam2()
        inference_state = predictor.init_state(video_path=folder_path)

        # Get initial point
        first_frame = cv2.imread(frame_paths[0])
        window_name = "Click on the squash ball"
        cv2.namedWindow(window_name)
        positive_points = []
        negative_points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                positive_points.append((x, y))
                # Draw red circle for positive point
                cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(window_name, display_frame)
            elif event == cv2.EVENT_RBUTTONDOWN:
                negative_points.append((x, y))
                # Draw blue circle for negative point
                cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)
                cv2.imshow(window_name, display_frame)

        # Get initial points
        first_frame = cv2.imread(frame_paths[0])
        display_frame = first_frame.copy()
        window_name = "Left click: Select ball | Right click: Select non-ball points | Press 's' to start"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Press 's' to start
                if not positive_points:
                    print("Please select at least one positive point before starting")
                    continue
                break
            elif key == ord('q'):  # Press 'q' to quit
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

        if not positive_points:
            raise ValueError("No points selected")

        processing_info["initial_positive_points"] = positive_points
        processing_info["initial_negative_points"] = negative_points

        # Combine positive and negative points
        points = np.array(positive_points + negative_points, dtype=np.float32)
        # 1 for positive points, 0 for negative points
        labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)

        with torch.cuda.amp.autocast():
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )

        print("Starting video propagation...")

        video_segments = {}
        with torch.cuda.amp.autocast():
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(inference_state):
                masks = [
                    (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i in range(len(out_obj_ids))
                ]
                if masks:
                    video_segments[out_frame_idx] = {
                        out_obj_id: mask for out_obj_id, mask in zip(out_obj_ids, masks)
                    }

                if out_frame_idx % 100 == 0:
                    torch.cuda.empty_cache()

        print("Creating output video...")

        height, width = first_frame.shape[:2]
        output_video_path = os.path.join(output_dir, "video", "ball_tracking.mp4")
        out = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        ball_positions = []
        ball_data = []  # For CSV export

        for frame_idx, masks in video_segments.items():
            frame = cv2.imread(frame_paths[frame_idx])

            for obj_id, mask in masks.items():
                if len(mask.shape) == 3:
                    mask = mask[0]

                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                mask_overlay = np.zeros_like(frame)
                mask_overlay[mask > 0] = [0, 0, 255]
                frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.5, 0)

                y_coords, x_coords = np.nonzero(mask)
                if len(y_coords) > 0:
                    center_y = int(np.mean(y_coords))
                    center_x = int(np.mean(x_coords))
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                    ball_positions.append((frame_idx, center_x, center_y))

                    # Store additional data for CSV
                    ball_data.append(
                        {
                            "frame": frame_idx,
                            "time": frame_idx / fps,
                            "x": center_x,
                            "y": center_y,
                            "mask_area": len(y_coords),
                        }
                    )

            if len(ball_positions) > 1:
                recent_positions = ball_positions[-10:]
                points = np.array(
                    [(pos[1], pos[2]) for pos in recent_positions], dtype=np.int32
                )
                cv2.polylines(frame, [points], False, (0, 255, 255), 2)

            cv2.putText(
                frame,
                f"Frame: {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            out.write(frame)

        out.release()

        # Save all data
        data_dir = os.path.join(output_dir, "data")

        # Save CSV
        csv_path = os.path.join(data_dir, "ball_tracking_data.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["frame", "time", "x", "y", "mask_area"]
            )
            writer.writeheader()
            writer.writerows(ball_data)

        # Save processing info
        processing_info["total_frames"] = len(ball_data)
        processing_info["tracking_duration"] = ball_data[-1]["time"] if ball_data else 0

        with open(os.path.join(data_dir, "processing_info.json"), "w") as f:
            json.dump(processing_info, f, indent=4)

        # Save trajectory plot
        if ball_positions:
            plt.figure(figsize=(15, 5))
            times = [d["time"] for d in ball_data]
            x_pos = [d["x"] for d in ball_data]
            y_pos = [d["y"] for d in ball_data]

            plt.subplot(121)
            plt.plot(times, x_pos)
            plt.title("Ball X Position over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("X Position (pixels)")

            plt.subplot(122)
            plt.plot(times, y_pos)
            plt.title("Ball Y Position over Time")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Y Position (pixels)")

            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, "trajectory_plot.png"))
            plt.close()

        print(f"Processing complete! Results saved in: {output_dir}")
        print(f"- Video: {output_video_path}")
        print(f"- Data: {csv_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        if is_video and os.path.exists("temp_frames"):
            shutil.rmtree("temp_frames")


if __name__ == "__main__":
    input_path = "farag v coll chopped 5sec.mp4"
    track_squash_ball(input_path)
