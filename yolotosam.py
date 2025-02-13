import cv2
import torch
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor
import numpy as np
import os
from datetime import datetime
import csv

def initialize_models():
    """Initialize both YOLO and SAM2 models"""
    # Initialize YOLO
    yolo_model = YOLO("trained-models/g-ball2(white_latest).pt")
    
    # Initialize SAM2
    sam2_checkpoint = "trained-models/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    sam2_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    return yolo_model, sam2_predictor

def process_video(input_path="video.mp4", yolo_interval=30, chunk_size=500):
    """
    Process video using both YOLO and SAM2
    yolo_interval: how often to run YOLO detection (in frames)
    chunk_size: number of frames to process at once
    """
    # Initialize models
    yolo_model, sam2_predictor = initialize_models()
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output directory and files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"tracking_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "tracked_video.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    csv_path = os.path.join(output_dir, "detections.csv")
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'x', 'y', 'confidence', 'tracking_method'])
    
    # Initialize SAM2 state
    inference_state = sam2_predictor.init_state()
    last_ball_position = None
    
    frame_count = 0
    chunk_frames = []
    
    print(f"Processing video with {total_frames} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        chunk_frames.append(frame)
        frame_count += 1
        
        # Process chunk when it reaches chunk_size or end of video
        if len(chunk_frames) == chunk_size or not ret:
            print(f"Processing frames {frame_count - len(chunk_frames)} to {frame_count}")
            
            for i, current_frame in enumerate(chunk_frames):
                current_frame_num = frame_count - len(chunk_frames) + i + 1
                
                # Run YOLO detection periodically
                if current_frame_num % yolo_interval == 0:
                    results = yolo_model(current_frame, verbose=False)
                    
                    if len(results[0].boxes) > 0:
                        # Get highest confidence detection
                        box = results[0].boxes[0]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Update SAM2 with new point
                        points = np.array([[center_x, center_y]], dtype=np.float32)
                        labels = np.array([1], dtype=np.int32)
                        
                        _, out_obj_ids, out_mask_logits = sam2_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=current_frame_num,
                            obj_id=1,
                            points=points,
                            labels=labels,
                        )
                        
                        last_ball_position = (center_x, center_y)
                        
                        # Write to CSV
                        csv_writer.writerow([
                            current_frame_num,
                            center_x, center_y,
                            float(box.conf.cpu().numpy()[0]),
                            'YOLO'
                        ])
                
                # Use SAM2 to track between YOLO detections
                if last_ball_position is not None:
                    with torch.cuda.amp.autocast():
                        out_frame_idx, out_obj_ids, out_mask_logits = next(
                            sam2_predictor.propagate_in_video(inference_state)
                        )
                        
                        if out_mask_logits is not None and len(out_mask_logits) > 0:
                            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                            y_coords, x_coords = np.nonzero(mask)
                            if len(y_coords) > 0:
                                center_x = int(np.mean(x_coords))
                                center_y = int(np.mean(y_coords))
                                last_ball_position = (center_x, center_y)
                                
                                # Write to CSV
                                csv_writer.writerow([
                                    current_frame_num,
                                    center_x, center_y,
                                    1.0,  # confidence for SAM2 tracking
                                    'SAM2'
                                ])
                
                # Draw tracking visualization
                if last_ball_position is not None:
                    cv2.circle(current_frame, last_ball_position, 5, (0, 255, 255), -1)
                
                writer.write(current_frame)
            
            # Clear chunk
            chunk_frames = []
            torch.cuda.empty_cache()
            
            # Print progress
            print(f"Progress: {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
    
    # Cleanup
    cap.release()
    writer.release()
    csv_file.close()
    
    print(f"\nProcessing complete!")
    print(f"Output video saved to: {output_path}")
    print(f"Detections saved to: {csv_path}")

if __name__ == "__main__":
    input_video = "C:/Users/default.DESKTOP-7FKFEEG/Downloads/farag v elshorbagy 2019 chopped.mp4"
    process_video(input_video, yolo_interval=30, chunk_size=500)