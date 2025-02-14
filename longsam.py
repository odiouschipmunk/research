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
from tqdm import tqdm

# Constants for chunking
CHUNK_SIZE = 300  # Reduced chunk size
OVERLAP_FRAMES = 20  # Reduced overlap
THREAD_WORKERS = 8  # Keep thread workers
BATCH_SIZE = 32  # Reduced batch size for better memory management

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

def extract_frames_efficiently(video_path, temp_folder="temp_frames", start_frame=0, num_frames=None):
    """Extract frames more efficiently using threading."""
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    # Use CUDA-accelerated video decoding if available
    if torch.cuda.is_available():
        # Use OpenCV's CUDA backend if available
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            cap.set(cv2.CAP_PROP_HW_DEVICE, 0)  # Use first CUDA device

    # Optimize video capture properties
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1024)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if num_frames is None:
        num_frames = total_frames - start_frame

    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def save_frame(args):
        frame, index = args
        if frame is not None:
            frame_path = os.path.join(temp_folder, f"{index + 1}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return frame_path
        return None

    frame_paths = []
    processed_frames = 0

    while processed_frames < num_frames:
        frames_batch = []
        indices_batch = []

        # Read frames in batches
        for _ in range(BATCH_SIZE):
            if processed_frames >= num_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frames_batch.append(frame)
            indices_batch.append(start_frame + processed_frames)
            processed_frames += 1

        if frames_batch:
            with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:
                futures = []
                for frame, idx in zip(frames_batch, indices_batch):
                    futures.append(executor.submit(save_frame, (frame, idx)))

                for i, future in enumerate(futures):
                    frame_paths.append(future.result())

    cap.release()
    return [p for p in frame_paths if p is not None], fps

def process_video_chunk(predictor, frame_paths, inference_state, start_frame, initial_points=None, initial_labels=None, prev_mask=None):
    """Process a chunk of video frames with tracking continuity."""
    with torch.amp.autocast('cuda'):
        if initial_points is not None and initial_labels is not None:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=initial_points,
                labels=initial_labels,
            )
        elif prev_mask is not None:
            # Ensure mask is 2D boolean array
            if len(prev_mask.shape) == 3:
                prev_mask = prev_mask[0]  # Take first channel if 3D
            prev_mask = prev_mask.astype(bool)
            
            try:
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    mask=prev_mask,
                )
            except AssertionError:
                print("Warning: Mask dimension error, attempting to recover...")
                # Try to recover using points from mask
                y_coords, x_coords = np.nonzero(prev_mask)
                if len(y_coords) > 0:
                    center_y = int(np.mean(y_coords))
                    center_x = int(np.mean(x_coords))
                    points = np.array([[center_x, center_y]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        points=points,
                        labels=labels,
                    )
                else:
                    raise RuntimeError("Could not recover from invalid mask")
        else:
            raise RuntimeError("Either initial points or previous mask must be provided")

        video_segments = {}

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            masks = [(out_mask_logits[i] > 0.0).cpu().numpy() for i in range(len(out_obj_ids))]
            if masks:
                video_segments[out_frame_idx + start_frame] = {
                    out_obj_id: mask for out_obj_id, mask in zip(out_obj_ids, masks)
                }

            if out_frame_idx % 100 == 0:
                torch.cuda.empty_cache()

    # Return the last mask for the next chunk
    last_frame_idx = max(video_segments.keys()) if video_segments else start_frame
    last_mask = next(iter(video_segments[last_frame_idx].values())) if video_segments else None

    return video_segments, last_mask

def track_squash_ball(input_path):
    try:
        output_dir = create_output_directory(input_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        processing_info = {
            "input_file": input_path,
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cuda_available": torch.cuda.is_available(),
        }

        # Get video information
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        processing_info["total_frames"] = total_frames
        processing_info["fps"] = fps

        # Initialize model
        predictor = initialize_sam2()

        # Get initial points from first frame
        first_chunk_paths, _ = extract_frames_efficiently(input_path, "temp_frames", 0, 1)
        first_frame = cv2.imread(first_chunk_paths[0])
        
        # Get user input for initial points (same as before)
        positive_points, negative_points = get_user_points(first_frame)
        points = np.array(positive_points + negative_points, dtype=np.float32)
        labels = np.array([1] * len(positive_points) + [0] * len(negative_points), dtype=np.int32)

        # Process video in chunks
        all_segments = {}
        prev_mask = None

        for chunk_start in tqdm(range(0, total_frames, CHUNK_SIZE - OVERLAP_FRAMES)):
            chunk_size = min(CHUNK_SIZE, total_frames - chunk_start)
            
            # Extract frames for current chunk
            chunk_paths, _ = extract_frames_efficiently(
                input_path, "temp_frames", chunk_start, chunk_size
            )
            
            # Initialize state for chunk
            inference_state = predictor.init_state(video_path=os.path.dirname(chunk_paths[0]))
            
            # Process chunk
            chunk_segments, prev_mask = process_video_chunk(
                predictor,
                chunk_paths,
                inference_state,
                chunk_start,
                points if chunk_start == 0 else None,
                labels if chunk_start == 0 else None,
                prev_mask if chunk_start > 0 else None
            )
            
            all_segments.update(chunk_segments)
            
            # Clean up
            shutil.rmtree("temp_frames")
            torch.cuda.empty_cache()

        # Create output video and save data
        create_output_video(all_segments, input_path, output_dir, fps, total_frames)
        save_tracking_data(all_segments, output_dir, fps)
        
        print(f"Processing complete! Results saved in: {output_dir}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        if os.path.exists("temp_frames"):
            shutil.rmtree("temp_frames")

def get_user_points(frame):
    """Get initial points from user input."""
    window_name = "Left click: Select ball | Right click: Select non-ball points | Press 's' to start"
    cv2.namedWindow(window_name)
    display_frame = frame.copy()
    positive_points = []
    negative_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            positive_points.append((x, y))
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, display_frame)
        elif event == cv2.EVENT_RBUTTONDOWN:
            negative_points.append((x, y))
            cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow(window_name, display_frame)

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if not positive_points:
                print("Please select at least one positive point before starting")
                continue
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return [], []

    cv2.destroyAllWindows()
    return positive_points, negative_points

def create_output_video(video_segments, input_path, output_dir, fps, total_frames):
    """Create output video with tracking visualization."""
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = os.path.join(output_dir, "video", "ball_tracking.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    ball_positions = []
    
    for frame_idx in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in video_segments:
            masks = video_segments[frame_idx]
            for obj_id, mask in masks.items():
                if len(mask.shape) == 3:
                    mask = mask[0]
                
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                mask_overlay = np.zeros_like(frame)
                mask_overlay[mask > 0] = [0, 0, 255]
                frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.5, 0)
                
                y_coords, x_coords = np.nonzero(mask)
                if len(y_coords) > 0:
                    center_y = int(np.mean(y_coords))
                    center_x = int(np.mean(x_coords))
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                    ball_positions.append((frame_idx, center_x, center_y))
        
        if len(ball_positions) > 1:
            recent_positions = ball_positions[-10:]
            points = np.array([(pos[1], pos[2]) for pos in recent_positions], dtype=np.int32)
            cv2.polylines(frame, [points], False, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    cap.release()
    out.release()

def save_tracking_data(video_segments, output_dir, fps):
    """Save tracking data to CSV and create visualizations."""
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    ball_data = []
    for frame_idx, masks in video_segments.items():
        for obj_id, mask in masks.items():
            y_coords, x_coords = np.nonzero(mask if len(mask.shape) == 2 else mask[0])
            if len(y_coords) > 0:
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))
                ball_data.append({
                    "frame": frame_idx,
                    "time": frame_idx / fps,
                    "x": center_x,
                    "y": center_y,
                    "mask_area": len(y_coords),
                })
    
    # Save CSV
    csv_path = os.path.join(data_dir, "ball_tracking_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time", "x", "y", "mask_area"])
        writer.writeheader()
        writer.writerows(ball_data)
    
    # Create and save trajectory plot
    if ball_data:
        create_trajectory_plot(ball_data, data_dir)

def create_trajectory_plot(ball_data, data_dir):
    """Create and save trajectory plots."""
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

if __name__ == "__main__":
    input_path = "C:\\Users\\default.DESKTOP-7FKFEEG\\Downloads\\farag v elshorbagy 2019 chopped.mp4"
    track_squash_ball(input_path)