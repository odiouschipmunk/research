import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
from PIL import Image

def initialize_sam2():
    """Initialize the SAM2 model."""
    sam2_checkpoint = "trained-models/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    return predictor

import natsort  # Ensure natsort is installed

def get_frame_paths(folder_path):
    """Get sorted list of frame paths with error handling."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder {folder_path} does not exist")

    frame_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    if not frame_files:
        raise ValueError(f"No images found in {folder_path}")

    return natsort.natsorted([os.path.join(folder_path, f) for f in frame_files])

def show_mask(mask, ax, random_color=False):
    """Visualize the mask."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1.0, 0.0, 0.0, 0.6])  # Red color for the ball
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=100):
    """Visualize the point prompts."""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
              s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
              s=marker_size, edgecolor='white', linewidth=1.25)

def track_squash_ball(folder_path):
    """Main function to track the squash ball."""
    # Initialize SAM2
    predictor = initialize_sam2()

    # Get frame paths
    frame_paths = get_frame_paths(folder_path)

    # Initialize inference state
    inference_state = predictor.init_state(video_path=folder_path)

    # Get initial ball position through user click
    first_frame = cv2.imread(frame_paths[0])
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(first_frame_rgb)
    plt.title("Click on the squash ball")
    point = plt.ginput(1)[0]
    plt.close()

    # Track the ball
    ball_id = 1  # Unique ID for the ball
    points = np.array([point], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    # Add initial point
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=ball_id,
        points=points,
        labels=labels,
    )

    # Show initial segmentation
    plt.figure(figsize=(12, 8))
    plt.imshow(first_frame_rgb)
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca())
    plt.title("Initial ball detection")
    plt.draw()  # Use draw instead of show to not block execution
    plt.pause(1)  # Pause briefly to show the plot

    print("Starting video propagation...")

    # Propagate through video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        print(f"Processing frame {out_frame_idx}")
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    print("Creating output visualizations...")

    # Visualize results
    os.makedirs("output_frames", exist_ok=True)

    # Store ball positions for trajectory
    ball_positions = []

    for frame_idx, masks in video_segments.items():
        frame = cv2.imread(frame_paths[frame_idx])
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(frame_rgb)

        for obj_id, mask in masks.items():
            show_mask(mask, ax)

            # Corrected ball center calculation
            mask_coords = np.nonzero(mask)  # Returns a tuple of arrays (y_coords, x_coords)
            if len(mask_coords[0]) > 0:  # If there are any True values in the mask
                center_y = int(np.mean(mask_coords[0]))
                center_x = int(np.mean(mask_coords[1]))
                ax.plot(center_x, center_y, 'yo', markersize=10)
                ball_positions.append((frame_idx, center_x, center_y))

        # Draw trajectory
        if len(ball_positions) > 1:
            trajectory_x = [pos[1] for pos in ball_positions[-10:]]  # Last 10 positions
            trajectory_y = [pos[2] for pos in ball_positions[-10:]]
            ax.plot(trajectory_x, trajectory_y, 'y-', linewidth=2, alpha=0.5)

        plt.title(f"Frame {frame_idx}")
        plt.savefig(f"output_frames/frame_{frame_idx:04d}.png")
        plt.close()

    print("Creating output video...")

    # Create video from output frames
    output_frames = sorted([f for f in os.listdir('output_frames') if f.endswith('.png')])
    if output_frames:
        frame = cv2.imread(os.path.join('output_frames', output_frames[0]))
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('ball_tracking_result.mp4', fourcc, 30.0, (width, height))

        for frame_name in output_frames:
            frame = cv2.imread(os.path.join('output_frames', frame_name))
            out.write(frame)

        out.release()

    print("Tracking complete! Results saved in 'ball_tracking_result.mp4'")

    # Show some statistics
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
        plt.show()

if __name__ == "__main__":
    folder_path = 'farag_v_elshorbagy_frames_small'
    track_squash_ball(folder_path)
