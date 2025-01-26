import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import psutil
import gc
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

def print_memory_usage():
    """Monitor memory usage for both RAM and CUDA."""
    process = psutil.Process()
    print(f"RAM Memory use: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"CUDA Memory use: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

def clear_memory():
    """Aggressive memory clearing."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    plt.close('all')
    gc.collect()

def process_frame_generator(video_path, max_size=None):
    """Generator that yields frames from video with optional resizing."""
    cap = cv2.VideoCapture(video_path)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize if specified
            if max_size:
                height, width = frame.shape[:2]
                if width > height:
                    if width > max_size:
                        scale = max_size / width
                        width = max_size
                        height = int(height * scale)
                else:
                    if height > max_size:
                        scale = max_size / height
                        height = max_size
                        width = int(width * scale)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            yield frame

    finally:
        cap.release()

def initialize_sam2():
    """Initialize the SAM2 model."""
    sam2_checkpoint = "trained-models/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    return predictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1.0, 0.0, 0.0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
              s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
              s=marker_size, edgecolor='white', linewidth=1.25)
def track_squash_ball(video_path, max_size=720, batch_size=10):
    """Memory-optimized ball tracking function."""
    print("Initializing SAM2...")
    predictor = initialize_sam2()

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # Create output video writer
    output_path = f"{os.path.splitext(video_path)[0]}_tracked.mp4"
    first_frame = next(process_frame_generator(video_path, max_size))
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Get initial ball position
    plt.figure(figsize=(12, 8))
    plt.imshow(first_frame)
    plt.title("Click on the squash ball")
    point = plt.ginput(1)[0]
    plt.close('all')

    # Initialize tracking
    ball_id = 1
    points = np.array([point], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    # Initialize inference state with video path
    inference_state = predictor.init_state(video_path=video_path)  # Fixed this line

    # Add initial point
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=ball_id,
        points=points,
        labels=labels,
    )

    # Track ball through video
    ball_positions = []
    frame_generator = process_frame_generator(video_path, max_size)

    try:
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame_idx, frame in enumerate(frame_generator):
                if frame_idx % batch_size == 0:
                    clear_memory()

                # Get next propagation
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    # Process frame
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(frame)

                    # Show mask and get ball position
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                    show_mask(mask, ax)

                    # Calculate ball center
                    mask_coords = np.nonzero(mask)
                    if len(mask_coords[0]) > 0:
                        center_y = int(np.mean(mask_coords[0]))
                        center_x = int(np.mean(mask_coords[1]))
                        ax.plot(center_x, center_y, 'yo', markersize=10)
                        ball_positions.append((frame_idx, center_x, center_y))

                    # Draw trajectory
                    if len(ball_positions) > 1:
                        last_positions = ball_positions[-10:]
                        trajectory_x = [pos[1] for pos in last_positions]
                        trajectory_y = [pos[2] for pos in last_positions]
                        ax.plot(trajectory_x, trajectory_y, 'y-', linewidth=2, alpha=0.5)

                    # Convert plot to image and write to video
                    plt.axis('off')
                    plt.tight_layout(pad=0)

                    # Convert plot to image
                    fig.canvas.draw()
                    plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plot_image = cv2.resize(plot_image, (width, height))

                    # Write to video
                    out.write(cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR))

                    plt.close('all')
                    break  # Process only one propagation per frame

                pbar.update(1)

    finally:
        out.release()
        clear_memory()

    # Plot trajectory analysis
    if ball_positions:
        plt.figure(figsize=(15, 5))
        frames, x_pos, y_pos = zip(*ball_positions)

        plt.subplot(121)
        plt.plot(frames, x_pos)
        plt.title('Ball X Position over Time')
        plt.xlabel('Frame')
        plt.ylabel('X Position')

        plt.subplot(122)
        plt.plot(frames, y_pos)
        plt.title('Ball Y Position over Time')
        plt.xlabel('Frame')
        plt.ylabel('Y Position')

        plt.savefig(f"{os.path.splitext(video_path)[0]}_trajectory.png")
        plt.close('all')

    print(f"Tracking complete! Results saved in '{output_path}'")

if __name__ == "__main__":
    video_path = 'input_video.mp4'  # Replace with your video file path
    track_squash_ball(video_path)
