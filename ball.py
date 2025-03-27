import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class KalmanFilter:
    def __init__(self, process_variance=0.1, measurement_variance=0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = np.zeros(4)  # [x, y, vx, vy]
        self.posteri_error_estimate = np.eye(4)
        
        # State transition matrix (constant velocity model)
        self.A = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * self.process_variance
        
        # Measurement noise covariance
        self.R = np.eye(2) * self.measurement_variance
        
        self.initialized = False
        
    def update(self, measurement):
        # Initialize if first measurement
        if not self.initialized and measurement is not None:
            self.posteri_estimate[0] = measurement[0]
            self.posteri_estimate[1] = measurement[1]
            self.initialized = True
            return self.posteri_estimate[0:2]
        
        # Predict step (a priori)
        priori_estimate = self.A @ self.posteri_estimate
        priori_error_estimate = self.A @ self.posteri_error_estimate @ self.A.T + self.Q
        
        # If no measurement, return prediction only
        if measurement is None:
            self.posteri_estimate = priori_estimate
            self.posteri_error_estimate = priori_error_estimate
            return self.posteri_estimate[0:2]
        
        # Update step (a posteriori)
        innovation = measurement - self.H @ priori_estimate
        innovation_covariance = self.H @ priori_error_estimate @ self.H.T + self.R
        kalman_gain = priori_error_estimate @ self.H.T @ np.linalg.inv(innovation_covariance)
        
        self.posteri_estimate = priori_estimate + kalman_gain @ innovation
        self.posteri_error_estimate = (np.eye(4) - kalman_gain @ self.H) @ priori_error_estimate
        
        return self.posteri_estimate[0:2]
    
    def get_velocity(self):
        return self.posteri_estimate[2:4]

def track_squash_ball(
    video_path, 
    model_path="trained-models/g-ball2(white_latest).pt", 
    conf_threshold=0.25,  # Slightly lower default threshold
    save_path=None,
    max_speed=120,  # Increased max speed for fast shots
    trail_length=20,  # Longer trail for better visualization
    max_frames_to_interpolate=8,  # Increased interpolation capacity
    use_kalman=True,  # Enable Kalman filtering
    process_variance=0.1,  # Kalman filter process noise
    measurement_variance=0.5  # Kalman filter measurement noise
):
    """
    Track a squash ball across a video using YOLO model with enhanced tracking.
    
    Args:
        video_path (str): Path to the input video file
        model_path (str): Path to the trained YOLO model
        conf_threshold (float): Confidence threshold for ball detection
        save_path (str): Path to save the output. If None, will create a directory in tracking_output/
        max_speed (int): Maximum allowed speed of the ball between frames (for filtering false positives)
        trail_length (int): Length of the trailing visualization
        max_frames_to_interpolate (int): Maximum frames to interpolate when ball is not detected
        use_kalman (bool): Whether to use Kalman filtering for smoother tracking
        process_variance (float): Kalman filter process noise parameter
        measurement_variance (float): Kalman filter measurement noise parameter
    
    Returns:
        tuple: (output_video_path, ball_positions_csv_path)
    """
    # Initialize YOLO model
    model = YOLO(model_path)
    # Force model to use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_path is None:
        output_dir = os.path.join("tracking_output", f"tracking_{timestamp}")
    else:
        output_dir = save_path
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize video writer
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Initialize CSV for saving ball positions
    ball_positions_csv_path = os.path.join(output_dir, "ball_positions.csv")
    csv_file = open(ball_positions_csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'time_sec', 'x', 'y', 'confidence', 'estimated', 'velocity_x', 'velocity_y'])
    
    # Initialize Kalman filter if enabled
    kalman = KalmanFilter(process_variance=process_variance, measurement_variance=measurement_variance) if use_kalman else None
    
    # Initialize data structures for trajectory
    ball_positions = []
    last_positions = []  # For displaying recent trajectory
    last_valid_position = None
    frames_since_detection = 0
    
    # Create court mask (assuming lower part of the frame is the court)
    # This can be adjusted based on the specific court layout
    court_mask = np.zeros((height, width), dtype=np.uint8)
    court_mask[height//4:, :] = 255  # Assume court is in the lower 3/4 of the frame
    
    # History of ball velocities for better filtering
    velocity_history = []
    max_velocity_history = 5
    
    # Process video frame by frame
    frame_number = 0
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        
        # Run YOLO detection
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Initialize ball position for current frame
        current_ball_position = None
        is_estimated = False
        velocity_x, velocity_y = 0, 0
        
        # Check if any ball was detected
        if results and len(results[0].boxes) > 0:
            # Get all ball detections sorted by confidence
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            
            # Get positions of all potential balls
            potential_balls = []
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(confidences[i])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Check if detection is in the court area (using mask)
                if center_y < height - 10:  # Avoid very bottom of frame
                    
                    # Calculate distance from last valid position (if available)
                    distance = float('inf')
                    expected_distance = float('inf')
                    
                    if last_valid_position is not None:
                        distance = np.sqrt((center_x - last_valid_position[0])**2 + 
                                          (center_y - last_valid_position[1])**2)
                        
                        # If we have velocity history, use it to predict expected position
                        if velocity_history:
                            avg_vx = np.mean([v[0] for v in velocity_history])
                            avg_vy = np.mean([v[1] for v in velocity_history])
                            expected_x = last_valid_position[0] + avg_vx
                            expected_y = last_valid_position[1] + avg_vy
                            expected_distance = np.sqrt((center_x - expected_x)**2 + 
                                                       (center_y - expected_y)**2)
                    
                    # Add to potential balls with weighted score
                    score = confidence * 0.5  # Base on confidence
                    
                    # If we have history, factor in expected position
                    if last_valid_position is not None:
                        if expected_distance < float('inf'):
                            position_score = max(0, 1 - (expected_distance / (max_speed * 1.5)))
                            score += position_score * 0.5
                        else:
                            position_score = max(0, 1 - (distance / max_speed))
                            score += position_score * 0.3
                    
                    potential_balls.append((center_x, center_y, confidence, distance, (x1, y1, x2, y2), score))
            
            # Choose the best ball detection
            if potential_balls:
                # Sort by combined score
                potential_balls.sort(key=lambda x: x[5], reverse=True)
                
                # Use the best detection
                best_ball = potential_balls[0]
                center_x, center_y, confidence, _, box_coords, _ = best_ball
                
                # Apply kalman filter if enabled
                if use_kalman:
                    filtered_pos = kalman.update(np.array([center_x, center_y]))
                    smooth_x, smooth_y = filtered_pos
                    
                    # Get estimated velocity
                    velocity = kalman.get_velocity()
                    velocity_x, velocity_y = velocity
                    
                    # Update position with filtered values
                    center_x, center_y = int(smooth_x), int(smooth_y)
                
                # Extract box coordinates
                x1, y1, x2, y2 = box_coords
                
                # Save position to list and CSV
                time_sec = frame_number / fps
                
                # Save position
                current_ball_position = (center_x, center_y)
                
                # Update tracking info
                if last_valid_position is not None:
                    # Calculate velocity
                    if not use_kalman:
                        velocity_x = center_x - last_valid_position[0]
                        velocity_y = center_y - last_valid_position[1]
                    
                    # Keep velocity history
                    velocity_history.append((velocity_x, velocity_y))
                    if len(velocity_history) > max_velocity_history:
                        velocity_history.pop(0)
                
                last_valid_position = current_ball_position
                frames_since_detection = 0
                
                # Update recent trajectory for display
                last_positions.append(current_ball_position)
                if len(last_positions) > trail_length:
                    last_positions.pop(0)
                
                # Save to CSV and positions list
                ball_positions.append((frame_number, time_sec, center_x, center_y, confidence, False, velocity_x, velocity_y))
                csv_writer.writerow([frame_number, time_sec, center_x, center_y, confidence, False, velocity_x, velocity_y])
                
                # Draw detection on frame
                # Draw box around ball
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                
                # Add confidence text near the ball
                cv2.putText(frame, f"{confidence:.2f}", (center_x + 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                
        else:
            # No detection in this frame
            frames_since_detection += 1
            
            # If we've recently seen the ball and we're using Kalman, we can predict
            if last_valid_position is not None and frames_since_detection <= max_frames_to_interpolate:
                if use_kalman:
                    # Predict using Kalman filter
                    predicted_pos = kalman.update(None)  # Update without measurement
                    pred_x, pred_y = predicted_pos
                    center_x, center_y = int(pred_x), int(pred_y)
                    
                    # Get estimated velocity
                    velocity = kalman.get_velocity()
                    velocity_x, velocity_y = velocity
                    
                    is_estimated = True
                else:
                    # Simple linear interpolation based on last velocity
                    if velocity_history:
                        avg_vx = np.mean([v[0] for v in velocity_history])
                        avg_vy = np.mean([v[1] for v in velocity_history])
                        center_x = int(last_valid_position[0] + avg_vx * frames_since_detection)
                        center_y = int(last_valid_position[1] + avg_vy * frames_since_detection)
                        velocity_x, velocity_y = avg_vx, avg_vy
                        is_estimated = True
                    else:
                        center_x, center_y = last_valid_position
                
                # Update position
                current_ball_position = (center_x, center_y)
                
                # Save to CSV with estimated flag
                time_sec = frame_number / fps
                ball_positions.append((frame_number, time_sec, center_x, center_y, 0.0, True, velocity_x, velocity_y))
                csv_writer.writerow([frame_number, time_sec, center_x, center_y, 0.0, True, velocity_x, velocity_y])
                
                # Update trail
                last_positions.append(current_ball_position)
                if len(last_positions) > trail_length:
                    last_positions.pop(0)
                
                # Show the estimated position (with a different color to indicate interpolation)
                cv2.circle(frame, (center_x, center_y), 5, (0, 165, 255), -1)  # Orange color for interpolated
                
                # Add text to indicate interpolation
                cv2.putText(frame, "Interpolated", (center_x + 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
        
        # Draw trail if we have positions
        if len(last_positions) > 1:
            # Create trail with color gradient
            for i in range(1, len(last_positions)):
                # Color fades based on position in trail (newer = brighter)
                alpha = i / len(last_positions)
                color = (0, int(255 * (1 - alpha)), 255)
                cv2.line(frame, last_positions[i-1], last_positions[i], color, 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add detection status
        if frames_since_detection == 0:
            status = "Detected"
            status_color = (0, 255, 0)  # Green
        elif frames_since_detection <= max_frames_to_interpolate:
            status = "Interpolated"
            status_color = (0, 165, 255)  # Orange
        else:
            status = "Lost"
            status_color = (0, 0, 255)  # Red
        
        cv2.putText(frame, f"Status: {status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                   1, status_color, 2, cv2.LINE_AA)
        
        # Write frame to output video
        out.write(frame)
        
        # Update progress every 100 frames
        if frame_number % 100 == 0:
            print(f"Progress: {frame_number}/{total_frames} frames ({(frame_number/total_frames)*100:.1f}%)")
        
        frame_number += 1
    
    # Release resources
    cap.release()
    out.release()
    csv_file.close()
    
    # Post-process trajectory to smooth out any noise
    smoothed_positions = post_process_trajectory(ball_positions, output_dir)
    
    # Generate trajectory plot
    if ball_positions:
        plot_path = generate_trajectory_plot(smoothed_positions or ball_positions, output_dir, width, height)
        print(f"Trajectory plot saved to: {plot_path}")
        
        # Generate heat map
        heatmap_path = generate_heatmap(smoothed_positions or ball_positions, output_dir, width, height)
        print(f"Heatmap saved to: {heatmap_path}")
    
    print(f"Processing complete!")
    print(f"Output video saved to: {output_video_path}")
    print(f"Ball positions saved to: {ball_positions_csv_path}")
    
    # Save trajectory data as numpy array for further analysis
    np_path = os.path.join(output_dir, "ball_trajectory.npy")
    np.save(np_path, np.array(smoothed_positions or ball_positions))
    print(f"Trajectory data saved to: {np_path}")
    
    return output_video_path, ball_positions_csv_path

def post_process_trajectory(ball_positions, output_dir):
    """Post-process the trajectory to smooth it and filter outliers"""
    if len(ball_positions) < 10:
        return None
    
    # Extract positions
    x_positions = np.array([pos[2] for pos in ball_positions])
    y_positions = np.array([pos[3] for pos in ball_positions])
    
    # Apply Savitzky-Golay filter to smooth trajectory
    # Window size must be odd and poly_order must be less than window_size
    window_size = min(15, len(x_positions) - 2)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 5:
        return None
        
    try:
        smoothed_x = savgol_filter(x_positions, window_size, 3)
        smoothed_y = savgol_filter(y_positions, window_size, 3)
        
        # Create smoothed positions
        smoothed_positions = []
        for i, pos in enumerate(ball_positions):
            smoothed_positions.append((
                pos[0],       # frame
                pos[1],       # time
                int(smoothed_x[i]),  # x
                int(smoothed_y[i]),  # y
                pos[4],       # confidence
                pos[5],       # estimated
                pos[6],       # velocity_x
                pos[7]        # velocity_y
            ))
        
        # Save smoothed trajectory to CSV
        smooth_csv_path = os.path.join(output_dir, "ball_positions_smoothed.csv")
        with open(smooth_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'time_sec', 'x', 'y', 'confidence', 'estimated', 'velocity_x', 'velocity_y'])
            for pos in smoothed_positions:
                writer.writerow(pos)
                
        return smoothed_positions
    except:
        print("Warning: Trajectory smoothing failed, using original trajectory.")
        return None

def generate_trajectory_plot(ball_positions, output_dir, width, height):
    """Generate and save a plot of the ball trajectory"""
    frames = [pos[0] for pos in ball_positions]
    times = [pos[1] for pos in ball_positions]
    x_positions = [pos[2] for pos in ball_positions]
    y_positions = [pos[3] for pos in ball_positions]
    confidences = [pos[4] for pos in ball_positions]
    estimated = [pos[5] for pos in ball_positions]
    
    plt.figure(figsize=(12, 10))
    
    # Main plot - trajectory in x,y space (court view)
    plt.subplot(2, 2, 1)
    
    # Plot detected points in one color and estimated in another
    detected_indices = [i for i, est in enumerate(estimated) if not est]
    estimated_indices = [i for i, est in enumerate(estimated) if est]
    
    # Plot detected points
    if detected_indices:
        detected_times = [times[i] for i in detected_indices]
        detected_x = [x_positions[i] for i in detected_indices]
        detected_y = [y_positions[i] for i in detected_indices]
        scatter1 = plt.scatter(detected_x, detected_y, c=detected_times, 
                              cmap='viridis', s=50, alpha=0.7, marker='o')
    
    # Plot estimated points
    if estimated_indices:
        estimated_times = [times[i] for i in estimated_indices]
        estimated_x = [x_positions[i] for i in estimated_indices]
        estimated_y = [y_positions[i] for i in estimated_indices]
        scatter2 = plt.scatter(estimated_x, estimated_y, c=estimated_times,
                              cmap='autumn', s=30, alpha=0.5, marker='x')
    
    plt.colorbar(label='Time (seconds)')
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Invert y-axis to match image coordinates
    plt.title('Squash Ball Trajectory (Court View)')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.grid(True, alpha=0.3)
    
    # X position over time
    plt.subplot(2, 2, 2)
    plt.plot(times, x_positions, 'r-', alpha=0.7)
    plt.title('X Position vs Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('X Position (pixels)')
    plt.grid(True, alpha=0.3)
    
    # Y position over time
    plt.subplot(2, 2, 3)
    plt.plot(times, y_positions, 'b-', alpha=0.7)
    plt.title('Y Position vs Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Y Position (pixels)')
    plt.grid(True, alpha=0.3)
    
    # Confidence plot
    plt.subplot(2, 2, 4)
    # Plot confidence for detected points only
    detected_confidences = [confidences[i] for i in detected_indices]
    detected_times = [times[i] for i in detected_indices]
    if detected_indices:
        plt.scatter(detected_times, detected_confidences, c='g', alpha=0.7, s=20)
        plt.plot(detected_times, detected_confidences, 'g-', alpha=0.5)
    plt.title('Detection Confidence vs Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Confidence')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "trajectory_analysis.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    return plot_path

def generate_heatmap(ball_positions, output_dir, width, height, bin_size=20):
    """Generate a heatmap of ball positions on the court"""
    # Extract positions
    x_positions = [pos[2] for pos in ball_positions]
    y_positions = [pos[3] for pos in ball_positions]
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    heatmap, xedges, yedges = np.histogram2d(
        x_positions, y_positions, 
        bins=[width//bin_size, height//bin_size],
        range=[[0, width], [0, height]]
    )
    
    # Smooth the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    
    # Plot the heatmap
    plt.imshow(heatmap.T, origin='upper', extent=[0, width, height, 0], 
               cmap='hot', interpolation='nearest')
    plt.colorbar(label='Ball presence frequency')
    plt.title('Squash Ball Heatmap')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, "ball_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    return heatmap_path

def analyze_ball_metrics(ball_positions, fps):
    """Analyze ball metrics such as speed, acceleration, bounce points, etc."""
    if len(ball_positions) < 2:
        return None
    
    # Calculate speeds and accelerations
    speeds = []
    accelerations = []
    
    x_positions = [pos[2] for pos in ball_positions]
    y_positions = [pos[3] for pos in ball_positions]
    times = [pos[1] for pos in ball_positions]
    velocities_x = [pos[6] for pos in ball_positions]
    velocities_y = [pos[7] for pos in ball_positions]
    
    # Calculate speeds using pre-computed velocities when available
    for i in range(len(ball_positions)):
        vx = velocities_x[i]
        vy = velocities_y[i]
        
        speed = np.sqrt(vx**2 + vy**2)
        speeds.append((times[i], speed))
    
    # Calculate accelerations
    for i in range(1, len(speeds)):
        ds = speeds[i][1] - speeds[i-1][1]
        dt = speeds[i][0] - speeds[i-1][0]
        
        if dt > 0:
            acceleration = ds / dt
            accelerations.append((speeds[i][0], acceleration))
    
    # Improved bounce detection
    bounce_points = []
    bounce_threshold = 5  # Minimum y-velocity change to consider a bounce
    
    # Look for bounce points by analyzing vertical velocity changes
    for i in range(3, len(ball_positions)-3):
        # Get vertical velocities (using smoothed velocities from a window)
        vy_prev = np.mean(velocities_y[i-3:i])
        vy_next = np.mean(velocities_y[i:i+3])
        
        # Check if vertical velocity changes from positive to negative
        # (in screen coordinates, positive y is downward)
        if vy_prev > bounce_threshold and vy_next < -bounce_threshold:
            # Additional check: y position should be in lower part of the frame (court area)
            if y_positions[i] > height * 0.4:  # Adjust based on court position
                bounce_points.append((times[i], x_positions[i], y_positions[i]))
    
    return {
        'speeds': speeds,
        'accelerations': accelerations,
        'bounce_points': bounce_points,
        'avg_speed': np.mean([s[1] for s in speeds]) if speeds else 0,
        'max_speed': max([s[1] for s in speeds]) if speeds else 0,
        'num_bounces': len(bounce_points)
    }

def main():
    # Set specific video and confidence threshold
    video_path = "farag_elshorbagy_1m_chopped.mp4"
    conf_threshold = 0.25  # Slightly lower threshold for better detection rate
    
    # Check if the video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return
    
    # Process the video
    print(f"Processing video: {video_path} with confidence threshold: {conf_threshold}")
    output_video, output_csv = track_squash_ball(
        video_path, 
        conf_threshold=conf_threshold,
        use_kalman=True,
        process_variance=0.1,
        measurement_variance=0.5,
        max_frames_to_interpolate=8
    )
    
    print("\nTracking Statistics:")
    # Read the CSV to get positions
    ball_positions = []
    with open(output_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            frame, time_sec, x, y, conf, estimated, vx, vy = row
            ball_positions.append((
                int(frame), 
                float(time_sec), 
                float(x), 
                float(y), 
                float(conf), 
                estimated.lower() == "true", 
                float(vx), 
                float(vy)
            ))
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Analyze metrics
    metrics = analyze_ball_metrics(ball_positions, fps)
    if metrics:
        print(f"  - Total detections: {sum(1 for pos in ball_positions if not pos[5])}")
        print(f"  - Interpolated positions: {sum(1 for pos in ball_positions if pos[5])}")
        print(f"  - Average speed: {metrics['avg_speed']:.2f} pixels/second")
        print(f"  - Maximum speed: {metrics['max_speed']:.2f} pixels/second")
        print(f"  - Detected bounces: {metrics['num_bounces']}")
    
    print(f"\nTo view the results, open the video: {output_video}")

if __name__ == "__main__":
    main()
