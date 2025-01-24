import cv2
import torch
import numpy as np
import pandas as pd
from segment_anything import sam_model_registry, SamPredictor
from datetime import datetime
import time
import os

# Configuration
CHECKPOINT_PATH = "sam2_h.pth"
CHUNK_LENGTH_MIN = 2  # Minutes per chunk
FRAME_SKIP = 2        # Process every nth frame
DOWNSCALE_FACTOR = 0.5  # Reduce resolution to save VRAM
USE_FP16 = True       # Use half-precision if supported

# Initialize SAM2 with memory optimization
sam = sam_model_registry["vit_h"](checkpoint=CHECKPOINT_PATH)
if USE_FP16:
    sam = sam.half()
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

def process_video_chunk(chunk_path):
    cap = cv2.VideoCapture(chunk_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    chunk_positions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Downscale frame
        small_frame = cv2.resize(frame, (0,0), fx=DOWNSCALE_FACTOR, fy=DOWNSCALE_FACTOR)
        
        # Convert to half-precision if enabled
        if USE_FP16:
            small_frame = small_frame.astype(np.float16)
        
        # Process with SAM2
        with torch.cuda.amp.autocast(enabled=USE_FP16):
            predictor.set_image(small_frame)
            input_point = np.array([[small_frame.shape[1]//2, small_frame.shape[0]//2]])
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=np.array([1]),
                multimask_output=True
            )
        
        # Find best mask
        best_mask = masks[np.argmax(scores)]
        if np.any(best_mask):
            y, x = np.where(best_mask)
            x_center = int(np.mean(x) / DOWNSCALE_FACTOR)  # Scale back to original coordinates
            y_center = int(np.mean(y) / DOWNSCALE_FACTOR)
            
            chunk_positions.append({
                'frame': frame_count,
                'timestamp': (frame_count * FRAME_SKIP) / fps,
                'x': x_center,
                'y': y_center,
                'confidence': float(scores.max())
            })

        # Clear memory between frames
        del masks, scores
        torch.cuda.empty_cache()

    cap.release()
    return chunk_positions

# Split video into chunks using FFmpeg (run this first)
def split_video(input_path, chunk_length_min):
    chunk_seconds = chunk_length_min * 60
    output_pattern = f"temp_chunk_%03d.mp4"
    os.system(f"ffmpeg -i {input_path} -c copy -f segment -segment_time {chunk_seconds} {output_pattern}")
    return sorted([f for f in os.listdir() if f.startswith("temp_chunk_")])

# Main processing workflow
video_path = "squash_video.mp4"
chunk_files = split_video(video_path, CHUNK_LENGTH_MIN)

all_positions = []
start_time = time.time()

try:
    for i, chunk_file in enumerate(chunk_files):
        print(f"Processing chunk {i+1}/{len(chunk_files)}: {chunk_file}")
        chunk_positions = process_video_chunk(chunk_file)
        all_positions.extend(chunk_positions)
        
        # Save intermediate results
        temp_df = pd.DataFrame(all_positions)
        temp_df.to_csv("partial_results.csv", index=False)
        
        # Clean up chunk file
        os.remove(chunk_file)
        torch.cuda.empty_cache()

finally:
    # Save final results
    final_df = pd.DataFrame(all_positions)
    output_filename = f"ball_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    final_df.to_csv(output_filename, index=False)
    print(f"Processing completed in {time.time()-start_time:.2f}s. Results saved to {output_filename}")