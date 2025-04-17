#!/usr/bin/env python3
"""
Integrated Squash Game Analysis System
This script provides a complete pipeline for analyzing squash games from video input.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from matplotlib.font_manager import FontProperties
from pathlib import Path
import json
import argparse
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Set, Union, Any, Deque
from dataclasses import dataclass
from collections import deque
from tqdm import tqdm
import pandas as pd
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not found. LLM-based analysis will be disabled.")

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('squash_analyzer')

# Type aliases for improved readability
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Vector = Tuple[float, float]  # x, y components
KeypointType = List[List[float]]  # [x, y, confidence] for each keypoint

# =============== Ball Tracking Classes and Functions ===============

class KalmanFilter:
    """Kalman filter implementation for smoother ball tracking"""
    def __init__(self, process_variance=0.03, measurement_variance=0.1, disappearance_threshold=5):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = np.zeros(4)  # [x, y, vx, vy]
        self.posteri_error_estimate = np.eye(4)
        self.disappearance_threshold = disappearance_threshold
        self.frames_since_last_detection = 0
        self.confidence = 0.0
        
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
        self.prediction_history = []
        
    def update(self, measurement, measurement_confidence=1.0):
        """
        Update the filter with a new measurement.
        
        Args:
            measurement: New position measurement or None if not available
            measurement_confidence: Confidence of the measurement (0.0-1.0)
            
        Returns:
            Estimated position (x, y)
        """
        # Initialize if first measurement
        if not self.initialized and measurement is not None:
            self.posteri_estimate[0] = measurement[0]
            self.posteri_estimate[1] = measurement[1]
            self.initialized = True
            self.confidence = measurement_confidence
            return self.posteri_estimate[0:2], self.confidence
        
        # Predict step (a priori)
        priori_estimate = self.A @ self.posteri_estimate
        priori_error_estimate = self.A @ self.posteri_error_estimate @ self.A.T + self.Q
        
        # If no measurement, return prediction only
        if measurement is None:
            self.frames_since_last_detection += 1
            
            # Decrease prediction confidence as we go without measurements
            self.confidence = max(0.0, self.confidence - (0.1 * self.frames_since_last_detection / self.disappearance_threshold))
            
            # Apply gravity effect for better prediction (slight increase in y velocity)
            if self.initialized:
                # Only apply gravity if ball is moving downward
                if self.posteri_estimate[3] > 0:
                    priori_estimate[3] += 0.2  # Slight gravity effect
                
            self.posteri_estimate = priori_estimate
            self.posteri_error_estimate = priori_error_estimate
            
            # Store prediction for future reference
            self.prediction_history.append((self.posteri_estimate[0:2], self.confidence))
            if len(self.prediction_history) > 10:
                self.prediction_history.pop(0)
                
            return self.posteri_estimate[0:2], self.confidence
        
        # Update step (a posteriori) - only if measurement confidence is high enough
        if measurement_confidence > 0.2:
            innovation = measurement - self.H @ priori_estimate
            innovation_covariance = self.H @ priori_error_estimate @ self.H.T + self.R
            kalman_gain = priori_error_estimate @ self.H.T @ np.linalg.inv(innovation_covariance)
            
            # Weight the innovation by the measurement confidence
            weighted_innovation = innovation * measurement_confidence
            self.posteri_estimate = priori_estimate + kalman_gain @ weighted_innovation
            self.posteri_error_estimate = (np.eye(4) - kalman_gain @ self.H) @ priori_error_estimate
            
            # Reset frames counter and update confidence
            self.frames_since_last_detection = 0
            self.confidence = measurement_confidence
        else:
            # Use prediction only if measurement confidence is too low
            self.posteri_estimate = priori_estimate
            self.posteri_error_estimate = priori_error_estimate
            self.frames_since_last_detection += 1
            self.confidence = max(0.0, self.confidence - 0.1)
        
        # Store prediction
        self.prediction_history.append((self.posteri_estimate[0:2], self.confidence))
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
            
        return self.posteri_estimate[0:2], self.confidence
    
    def get_velocity(self):
        return self.posteri_estimate[2:4]
        
    def reset_if_lost(self):
        """Reset the filter if tracking is likely lost"""
        if self.frames_since_last_detection > self.disappearance_threshold:
            self.initialized = False
            self.frames_since_last_detection = 0
            self.confidence = 0.0
            self.prediction_history = []

# =============== Player Tracking Classes ===============

@dataclass
class Detection:
    """Dataclass to store detection information"""
    bbox: BBox
    confidence: float
    keypoints: Optional[KeypointType] = None
    track_id: Optional[int] = None
    
    @property
    def center(self) -> Point:
        """Calculate center of bounding box"""
        return ((self.bbox[0] + self.bbox[2]) / 2, 
                (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def height(self) -> float:
        """Calculate height of bounding box"""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def width(self) -> float:
        """Calculate width of bounding box"""
        return self.bbox[2] - self.bbox[0]
    
    def extract_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract region of interest from frame"""
        if frame is None or self.width <= 0 or self.height <= 0:
            return None
            
        x1, y1, x2, y2 = map(int, self.bbox)
        # Ensure bbox is within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return frame[y1:y2, x1:x2]

class PlayerTracker:
    """Tracks a player across video frames using multiple cues"""
    def __init__(
        self, 
        player_id: int, 
        max_history: int = 30, 
        appearance_history_size: int = 10,
        heatmap_resolution: Tuple[int, int] = (20, 20),
        motion_smoothing_factor: float = 0.7
    ):
        self.player_id = player_id
        self.track_id: Optional[int] = None
        
        # Position and motion tracking
        self.positions: Deque[Point] = deque(maxlen=max_history)
        self.keypoints_history: Deque[KeypointType] = deque(maxlen=max_history)
        self.long_term_positions: Deque[Point] = deque(maxlen=200)  # Increased for better trajectory modeling
        self.velocity: Vector = (0, 0)
        self.acceleration: Vector = (0, 0)
        self.motion_smoothing_factor = motion_smoothing_factor
        
        # Current state
        self.confidence: float = 0
        self.bbox: Optional[BBox] = None
        self.keypoints: Optional[KeypointType] = None
        self.missing_frames: int = 0
        self.last_reliable_position: Optional[Point] = None
        self.last_height: Optional[float] = None
        self.id_confidence: float = 1.0
        
        # Visual appearance modeling - improved
        self.appearance_history: Deque[np.ndarray] = deque(maxlen=appearance_history_size)
        self.appearance_features: Deque[np.ndarray] = deque(maxlen=appearance_history_size)
        
        # Set player color (Red for P1, Blue for P2)
        self.color: Tuple[int, int, int] = (0, 0, 255) if player_id == 1 else (255, 0, 0)
        
        # Position heatmap for spatial consistency checking
        self.position_heatmap: Optional[np.ndarray] = None
        self.heatmap_resolution = heatmap_resolution
        self.heatmap_width: Optional[int] = None
        self.heatmap_height: Optional[int] = None
        self.heatmap_updates: int = 0
        
        # Frame dimensions (will be set by tracking manager)
        self.frame_dimensions: Optional[Tuple[int, int]] = None
        
        # Velocity-based prediction for occlusion
        self.kalman = KalmanFilter(process_variance=0.01, measurement_variance=0.1)
        
        # Movement behavior model - helps with identity preservation
        self.court_position_preference = np.zeros((3, 2))  # (front/middle/back, left/right)
        
    def update_position_preference(self, region: str, side: str) -> None:
        """Update player's court position preference model"""
        region_idx = {'Front': 0, 'Middle': 1, 'Back': 2}.get(region, 1)
        side_idx = {'Left': 0, 'Right': 1}.get(side, 0)
        
        # Increment count for this region/side
        self.court_position_preference[region_idx, side_idx] += 1
    
    def get_position_similarity_score(self, region: str, side: str) -> float:
        """Calculate how well a position matches this player's movement patterns"""
        if np.sum(self.court_position_preference) < 10:
            return 0.5  # Not enough data yet
            
        region_idx = {'Front': 0, 'Middle': 1, 'Back': 2}.get(region, 1)
        side_idx = {'Left': 0, 'Right': 1}.get(side, 0)
        
        # Calculate probability this player would be in this region/side
        total = np.sum(self.court_position_preference)
        if total == 0:
            return 0.5
            
        region_prob = self.court_position_preference[region_idx, side_idx] / total
        return region_prob

    @property
    def center(self) -> Optional[Point]:
        """Get the center point of the player's bounding box"""
        if self.bbox is None:
            return None
        return ((self.bbox[0] + self.bbox[2]) / 2, 
                (self.bbox[1] + self.bbox[3]) / 2)

    def compute_appearance_features(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Compute simple appearance features from player ROI"""
        if roi is None or roi.size == 0:
            return None
            
        try:
            # Resize for consistency
            resized = cv2.resize(roi, (64, 128))
            
            # Compute histogram features
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            
            # Color histogram (focus on hue and saturation for clothing color)
            hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            
            # Normalize histograms
            cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
            
            # Combine features
            features = np.concatenate([hist_h.flatten(), hist_s.flatten()])
            return features
        except Exception as e:
            logger.warning(f"Error computing appearance features: {e}")
            return None
    
    def compare_appearance(self, features: np.ndarray) -> float:
        """Compare appearance features to history, return similarity score"""
        if not self.appearance_features or features is None:
            return 0.5  # Neutral score if no history
            
        scores = []
        for hist_features in self.appearance_features:
            if hist_features is not None:
                # Compute correlation between histograms
                score = cv2.compareHist(hist_features, features, cv2.HISTCMP_CORREL)
                scores.append(max(0, score))  # Ensure non-negative
                
        return np.mean(scores) if scores else 0.5

    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Set frame dimensions for heatmap calculations"""
        self.frame_dimensions = (width, height)

    def predict_position(self) -> Optional[Point]:
        """Predict player position based on motion model"""
        if not self.positions:
            return None
            
        # Use Kalman filter for prediction
        if len(self.positions) >= 2:
            last_pos = self.positions[-1]
            try:
                kalman_pred, conf = self.kalman.update(last_pos)
                # Verify the prediction is valid
                if conf > 0 and np.all(np.isfinite(kalman_pred)):
                    return (float(kalman_pred[0]), float(kalman_pred[1]))
                else:
                    return None
            except Exception as e:
                logger.warning(f"Error in player position prediction: {e}")
                return None
        
        # Fallback to last position if we can't predict
        last_pos = self.positions[-1]
        if last_pos is not None and len(last_pos) == 2:
            return last_pos
        return None

    def update_state(self, detection: Detection, frame: np.ndarray, 
                    court_region: str, court_side: str) -> None:
        """
        Update player state with new detection
        
        Args:
            detection: New detection for this player
            frame: Current video frame
            court_region: Region of court (Front/Middle/Back)
            court_side: Side of court (Left/Right)
        """
        # Update basic state
        self.bbox = detection.bbox
        self.keypoints = detection.keypoints
        self.confidence = detection.confidence
        
        # Update position history
        center = detection.center
        self.positions.append(center)
        self.long_term_positions.append(center)
        
        # Update player position in Kalman filter
        self.kalman.update(center, self.confidence)
        
        # Update keypoints history
        if detection.keypoints is not None:
            self.keypoints_history.append(detection.keypoints)
        
        # Update appearance history
        roi = detection.extract_roi(frame)
        if roi is not None:
            self.appearance_history.append(roi)
            features = self.compute_appearance_features(roi)
            if features is not None:
                self.appearance_features.append(features)
        
        # Update velocity and acceleration
        if len(self.positions) > 1:
            prev_pos = self.positions[-2]
            curr_velocity = (
                center[0] - prev_pos[0],
                center[1] - prev_pos[1]
            )
            
            # Smooth velocity
            self.velocity = (
                self.motion_smoothing_factor * self.velocity[0] + 
                (1 - self.motion_smoothing_factor) * curr_velocity[0],
                self.motion_smoothing_factor * self.velocity[1] + 
                (1 - self.motion_smoothing_factor) * curr_velocity[1]
            )
            
            # Update acceleration if we have previous velocity
            if len(self.positions) > 2:
                prev_prev_pos = self.positions[-3]
                prev_velocity = (
                    prev_pos[0] - prev_prev_pos[0],
                    prev_pos[1] - prev_prev_pos[1]
                )
                self.acceleration = (
                    self.velocity[0] - prev_velocity[0],
                    self.velocity[1] - prev_velocity[1]
                )
        
        # Update position heatmap
        self._update_heatmap(center)
        
        # Update court position preference model
        self.update_position_preference(court_region, court_side)
        
        # Reset missing frames counter
        self.missing_frames = 0
        
        # Update last reliable position
        if self.confidence > 0.5:
            self.last_reliable_position = center
            self.last_height = detection.height

    def mark_missing(self) -> None:
        """Mark player as missing in current frame"""
        self.missing_frames += 1
        self.bbox = None
        self.keypoints = None
        self.confidence = 0
        
        # Predict next position using Kalman filter
        if self.positions:
            pred, _ = self.kalman.update(None)
            if np.all(np.isfinite(pred)):
                predicted_pos = (float(pred[0]), float(pred[1]))
                self.positions.append(predicted_pos)

    def _update_heatmap(self, position: Point) -> None:
        """Update position heatmap"""
        if self.position_heatmap is None or self.frame_dimensions is None:
            self.heatmap_width = self.heatmap_resolution[0]
            self.heatmap_height = self.heatmap_resolution[1]
            self.position_heatmap = np.zeros((self.heatmap_height, self.heatmap_width))
        
        # Convert position to heatmap coordinates
        x = int(position[0] * self.heatmap_width / self.frame_dimensions[0])
        y = int(position[1] * self.heatmap_height / self.frame_dimensions[1])
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, self.heatmap_width - 1))
        y = max(0, min(y, self.heatmap_height - 1))
        
        # Update heatmap
        self.position_heatmap[y, x] += 1
        self.heatmap_updates += 1

class PlayerTrackingManager:
    """Manages tracking of multiple players, handling occlusions and identity switches"""
    def __init__(
        self, 
        num_players: int = 2,
        max_missing_frames: int = 30,
        position_consistency_check_interval: int = 60, 
        swap_confidence_threshold: float = 0.85
    ):
        self.players = {
            i+1: PlayerTracker(player_id=i+1) for i in range(num_players)
        }
        self.max_missing_frames = max_missing_frames
        self.position_consistency_check_interval = position_consistency_check_interval
        self.swap_confidence_threshold = swap_confidence_threshold
        self.initial_positions = {}
        self.frame_number = 0
        self.frame_dimensions = None
        
    def _get_court_region(self, y: float, height: int) -> str:
        """Determine court region based on y position"""
        if y < height * 0.33:
            return "Front"
        elif y < height * 0.66:
            return "Middle"
        else:
            return "Back"
    
    def _get_court_side(self, x: float, width: int) -> str:
        """Determine court side based on x position"""
        return "Left" if x < width / 2 else "Right"

    def assign_detections_to_players(self, detections: List[Detection], frame: np.ndarray) -> None:
        """
        Assign detections to players based on position and appearance
        
        Args:
            detections: List of detections from current frame
            frame: Current video frame
        """
        # If no detections, mark all players as missing
        if not detections:
            for player in self.players.values():
                player.mark_missing()
            return
        
        # Calculate multiple similarity metrics for assignment
        similarity_scores = {}
        for player_id, player in self.players.items():
            # Skip if player has no history yet
            if player.positions and len(player.positions) > 0:
                # Get predicted position from motion model
                player_predicted_pos = player.predict_position()
                
                # Calculate metrics for each detection
                for i, det in enumerate(detections):
                    det_center = det.center
                    
                    # 1. Position-based distance score (inversely proportional to distance)
                    if player_predicted_pos is not None and np.all(np.isfinite(player_predicted_pos)):
                        pos_dist = np.sqrt((player_predicted_pos[0] - det_center[0])**2 + 
                                        (player_predicted_pos[1] - det_center[1])**2)
                        
                        # Normalize distance (closer = higher score)
                        max_dist = np.sqrt(self.frame_dimensions[0]**2 + self.frame_dimensions[1]**2)
                        position_score = 1.0 - min(1.0, pos_dist / (max_dist/2))
                    else:
                        position_score = 0.5  # Neutral if no position history
                    
                    # 2. Appearance similarity score
                    appearance_score = 0.5  # Default neutral
                    roi = det.extract_roi(frame)
                    if roi is not None:
                        features = player.compute_appearance_features(roi)
                        if features is not None:
                            appearance_score = player.compare_appearance(features)
                    
                    # 3. Court region/side consistency score
                    region = self._get_court_region(det_center[1], self.frame_dimensions[1])
                    side = self._get_court_side(det_center[0], self.frame_dimensions[0])
                    position_preference_score = player.get_position_similarity_score(region, side)
                    
                    # 4. Movement consistency score - how well the detection matches player's velocity
                    movement_score = 0.5  # Default neutral
                    if len(player.positions) >= 2:
                        # Calculate expected position based on velocity
                        last_pos = player.positions[-1]
                        if last_pos is not None and len(last_pos) == 2 and all(np.isfinite(v) for v in player.velocity):
                            expected_pos = (last_pos[0] + player.velocity[0], last_pos[1] + player.velocity[1])
                            
                            # Calculate how well the detection matches the expected position
                            exp_dist = np.sqrt((expected_pos[0] - det_center[0])**2 + 
                                            (expected_pos[1] - det_center[1])**2)
                            movement_score = 1.0 - min(1.0, exp_dist / (max_dist/3))
                    
                    # Combine scores with different weights
                    combined_score = (
                        0.5 * position_score +         # Position is most important
                        0.25 * appearance_score +      # Appearance helps with identity preservation
                        0.15 * position_preference_score +  # Region preference for consistency
                        0.1 * movement_score           # Movement pattern consistency
                    )
                    
                    similarity_scores[(player_id, i)] = combined_score
            else:
                # For new players with no history, just use proximity to last known player positions
                for i, det in enumerate(detections):
                    det_center = det.center
                    similarity_scores[(player_id, i)] = 0.5  # Neutral score
        
        # If we have no similarity scores (first frame), use simple assignment
        if not similarity_scores:
            for i, det in enumerate(detections):
                if i < len(self.players):
                    player_id = i + 1
                    # Determine court region and side
                    det_center = det.center
                    region = self._get_court_region(det_center[1], self.frame_dimensions[1]) 
                    side = self._get_court_side(det_center[0], self.frame_dimensions[0])
                    # Update player state
                    self.players[player_id].update_state(det, frame, region, side)
            return
        
        # Use Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            
            # Create cost matrix (negative similarity for minimization)
            player_ids = list(self.players.keys())
            det_indices = list(range(len(detections)))
            
            cost_matrix = np.ones((len(player_ids), len(det_indices)))
            
            for (pid, did), score in similarity_scores.items():
                player_idx = player_ids.index(pid)
                if player_idx < len(player_ids) and did < len(det_indices):
                    # Convert similarity to cost (higher similarity = lower cost)
                    cost_matrix[player_idx, did] = 1.0 - score
            
            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Update player states with assigned detections
            assigned_dets = set()
            for pid_idx, det_idx in zip(row_ind, col_ind):
                if pid_idx < len(player_ids) and det_idx < len(detections):
                    player_id = player_ids[pid_idx]
                    det = detections[det_idx]
                    
                    # Determine court region and side
                    det_center = det.center
                    region = self._get_court_region(det_center[1], self.frame_dimensions[1])
                    side = self._get_court_side(det_center[0], self.frame_dimensions[0])
                    
                    # Only update if similarity is high enough
                    score = 1.0 - cost_matrix[pid_idx, det_idx]
                    if score >= 0.3:  # Threshold for assignment
                        # Update player state
                        self.players[player_id].update_state(det, frame, region, side)
                        assigned_dets.add(det_idx)
                    else:
                        # Mark as missing if similarity is too low
                        self.players[player_id].mark_missing()
            
            # Mark unassigned players as missing
            for player_id in player_ids:
                if player_id not in [player_ids[i] for i in row_ind]:
                    self.players[player_id].mark_missing()
                    
        except Exception as e:
            logger.warning(f"Error in detection assignment: {e}")
            # Fallback to simpler assignment
            for i, det in enumerate(detections):
                if i < len(self.players):
                    player_id = i + 1
                    # Determine court region and side
                    det_center = det.center
                    region = self._get_court_region(det_center[1], self.frame_dimensions[1])
                    side = self._get_court_side(det_center[0], self.frame_dimensions[0])
                    # Update player state
                    self.players[player_id].update_state(det, frame, region, side)
    
    def check_for_id_swaps(self, frame: np.ndarray) -> None:
        """
        Check for and correct player ID swaps based on position history and appearance
        
        Args:
            frame: Current video frame
        """
        # Only check for swaps if we have exactly 2 players
        if len(self.players) != 2 or 1 not in self.players or 2 not in self.players:
            return
            
        player1 = self.players[1]
        player2 = self.players[2]
        
        # Only proceed if both players have bounding boxes
        if player1.bbox is None or player2.bbox is None:
            return
            
        # Calculate position-based swap score
        position_swap_score = 0
        
        # 1. Check long-term position consistency
        if len(player1.long_term_positions) > 30 and len(player2.long_term_positions) > 30:
            try:
                # Calculate centroids of player movement areas
                p1_positions = [p for p in player1.long_term_positions if p is not None and len(p) == 2]
                p2_positions = [p for p in player2.long_term_positions if p is not None and len(p) == 2]
                
                if len(p1_positions) > 0 and len(p2_positions) > 0:
                    p1_center_x = np.mean([p[0] for p in p1_positions])
                    p1_center_y = np.mean([p[1] for p in p1_positions])
                    p2_center_x = np.mean([p[0] for p in p2_positions])
                    p2_center_y = np.mean([p[1] for p in p2_positions])
                    
                    # Current positions
                    p1_curr = player1.center
                    p2_curr = player2.center
                    
                    # Calculate distances if both positions are valid
                    if p1_curr is not None and p2_curr is not None:
                        # Current assignment distances
                        d1 = np.sqrt((p1_curr[0] - p1_center_x)**2 + (p1_curr[1] - p1_center_y)**2)
                        d2 = np.sqrt((p2_curr[0] - p2_center_x)**2 + (p2_curr[1] - p2_center_y)**2)
                        current_dist = d1 + d2
                        
                        # Swapped assignment distances
                        d1s = np.sqrt((p2_curr[0] - p1_center_x)**2 + (p2_curr[1] - p1_center_y)**2)
                        d2s = np.sqrt((p1_curr[0] - p2_center_x)**2 + (p1_curr[1] - p2_center_y)**2)
                        swapped_dist = d1s + d2s
                        
                        # If swapped distance is significantly smaller, it's likely an ID swap
                        if swapped_dist < current_dist * 0.8:
                            position_swap_score += 1
            except Exception as e:
                logger.warning(f"Error in position consistency check: {e}")
        
        # 2. Check court region consistency
        p1_center = player1.center
        p2_center = player2.center
        if p1_center is not None and p2_center is not None:
            try:
                p1_region = self._get_court_region(p1_center[1], self.frame_dimensions[1])
                p1_side = self._get_court_side(p1_center[0], self.frame_dimensions[0])
                p2_region = self._get_court_region(p2_center[1], self.frame_dimensions[1])
                p2_side = self._get_court_side(p2_center[0], self.frame_dimensions[0])
                
                # Check how well current positions match the players' preferred regions
                p1_score = player1.get_position_similarity_score(p1_region, p1_side)
                p2_score = player2.get_position_similarity_score(p2_region, p2_side)
                current_region_score = p1_score + p2_score
                
                # Check swapped scores
                p1s_score = player1.get_position_similarity_score(p2_region, p2_side)
                p2s_score = player2.get_position_similarity_score(p1_region, p1_side)
                swapped_region_score = p1s_score + p2s_score
                
                # If swapped scores are significantly better
                if swapped_region_score > current_region_score * 1.5:
                    position_swap_score += 1
            except Exception as e:
                logger.warning(f"Error in region consistency check: {e}")
        
        # 3. Check appearance consistency (if we have history)
        appearance_swap_score = 0
        if len(player1.appearance_features) > 0 and len(player2.appearance_features) > 0:
            try:
                # Extract ROIs from current frame
                p1_roi = None
                p2_roi = None
                if player1.bbox and player2.bbox:
                    p1_roi = frame[int(player1.bbox[1]):int(player1.bbox[3]), 
                                  int(player1.bbox[0]):int(player1.bbox[2])]
                    p2_roi = frame[int(player2.bbox[1]):int(player2.bbox[3]), 
                                  int(player2.bbox[0]):int(player2.bbox[2])]
                
                if p1_roi is not None and p2_roi is not None and p1_roi.size > 0 and p2_roi.size > 0:
                    # Compute features
                    p1_features = player1.compute_appearance_features(p1_roi)
                    p2_features = player2.compute_appearance_features(p2_roi)
                    
                    if p1_features is not None and p2_features is not None:
                        # Compare current assignment
                        p1_self_score = player1.compare_appearance(p1_features)
                        p2_self_score = player2.compare_appearance(p2_features)
                        current_app_score = p1_self_score + p2_self_score
                        
                        # Compare swapped assignment
                        p1_other_score = player1.compare_appearance(p2_features)
                        p2_other_score = player2.compare_appearance(p1_features)
                        swapped_app_score = p1_other_score + p2_other_score
                        
                        # If swapped appearance is a better match
                        if swapped_app_score > current_app_score * 1.2:
                            appearance_swap_score += 1
            except Exception as e:
                logger.warning(f"Error in appearance consistency check: {e}")
        
        # Make a decision based on combined evidence
        total_swap_score = position_swap_score + appearance_swap_score
        
        # If there's strong evidence of a swap, correct it
        if total_swap_score >= 2:
            logger.info(f"Detected ID swap at frame {self.frame_number}, correcting...")
            # Swap player trackers
            self.players[1], self.players[2] = self.players[2], self.players[1]
            # Update player IDs
            self.players[1].player_id = 1
            self.players[2].player_id = 2

    def update(self, results: Any, frame: np.ndarray) -> None:
        """
        Update player tracking with new detections
        
        Args:
            results: YOLO detection results
            frame: Current video frame
        """
        # Store frame dimensions if not set
        if self.frame_dimensions is None:
            self.frame_dimensions = (frame.shape[1], frame.shape[0])
            # Set frame dimensions for all players
            for player in self.players.values():
                player.set_frame_dimensions(*self.frame_dimensions)
        
        # Process detections
        detections = []
        if results and len(results.boxes) > 0:
            boxes = results.boxes
            keypoints = results.keypoints
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                
                # Get keypoints if available
                kpts = None
                if keypoints is not None and i < len(keypoints):
                    try:
                        # Handle Ultralytics Keypoints class specifically
                        if hasattr(keypoints[i], 'xy') and hasattr(keypoints[i], 'conf'):
                            # This is an Ultralytics Keypoints object
                            kpts_xy = keypoints[i].xy.cpu().numpy()
                            conf = keypoints[i].conf.cpu().numpy()
                            
                            # Create array with [x, y, conf] format
                            # Determine number of keypoints carefully
                            if len(kpts_xy.shape) == 3:  # shape like (1, 17, 2)
                                num_keypoints = kpts_xy.shape[1]
                                kpts_xy = kpts_xy.reshape(num_keypoints, 2)
                            elif len(kpts_xy.shape) == 2:  # shape like (17, 2)
                                num_keypoints = kpts_xy.shape[0]
                            else:
                                # Handle unexpected shape
                                logger.warning(f"Unexpected keypoints xy shape: {kpts_xy.shape}")
                                # Try to infer from conf shape if possible
                                if hasattr(conf, 'shape') and len(conf.shape) > 0:
                                    if len(conf.shape) == 1:  # shape like (17,)
                                        num_keypoints = conf.shape[0]
                                    elif len(conf.shape) == 2:  # shape like (1, 17)
                                        num_keypoints = conf.shape[1]
                                    else:
                                        # Last resort
                                        num_keypoints = kpts_xy.size // 2
                                else:
                                    # Last resort
                                    num_keypoints = kpts_xy.size // 2
                            
                            kpts = np.zeros((num_keypoints, 3))
                            
                            try:
                                # Ensure kpts_xy has the right shape
                                if kpts_xy.shape != (num_keypoints, 2):
                                    kpts_xy = kpts_xy.reshape(num_keypoints, 2)
                                
                                # Assign coordinates
                                kpts[:, 0:2] = kpts_xy
                                
                                # Ensure conf has the right shape
                                if hasattr(conf, 'shape'):
                                    if len(conf.shape) > 1:
                                        conf = conf.reshape(num_keypoints)
                                    # Assign confidence
                                    kpts[:, 2] = conf
                                else:
                                    kpts[:, 2] = 0.9  # Default high confidence
                            except Exception as e:
                                logger.error(f"Failed to assign keypoints xy to output array: {e}")
                                # Fallback: copy element by element
                                for j in range(min(num_keypoints, kpts_xy.shape[0])):
                                    if j < kpts_xy.shape[0] and kpts_xy.shape[1] >= 2:
                                        kpts[j, 0] = kpts_xy[j, 0]
                                        kpts[j, 1] = kpts_xy[j, 1]
                                    if j < len(conf):
                                        kpts[j, 2] = conf[j]
                        # Fall back to previous methods if not an Ultralytics Keypoints object
                        elif isinstance(keypoints[i], np.ndarray):
                            kpts = keypoints[i]
                        elif hasattr(keypoints[i], 'cpu'):
                            cpu_keypoints = keypoints[i].cpu()
                            if hasattr(cpu_keypoints, 'numpy'):
                                kpts = cpu_keypoints.numpy()
                            elif hasattr(cpu_keypoints, 'data'):
                                kpts = cpu_keypoints.data.numpy()
                        else:
                            logger.warning(f"Unsupported keypoints format: {type(keypoints[i])}")
                            logger.warning(f"Available attributes: {dir(keypoints[i])}")
                    except Exception as e:
                        logger.warning(f"Error converting keypoints: {e}")
                        logger.warning(f"Keypoints type: {type(keypoints[i])}")
                        logger.warning(f"Available attributes: {dir(keypoints[i])}")
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    keypoints=kpts
                ))
        
        # Update tracking
        self.assign_detections_to_players(detections, frame)
        
        # Check for ID swaps periodically
        if self.frame_number % self.position_consistency_check_interval == 0:
            self.check_for_id_swaps(frame)
        
        # Mark missing players
        for player in self.players.values():
            if player.bbox is None:
                player.mark_missing()
        
        self.frame_number += 1

    def visualize_tracking(self, frame: np.ndarray) -> np.ndarray:
        """Visualize player tracking on frame"""
        display_frame = frame.copy()
        
        for player in self.players.values():
            if player.bbox is not None:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, player.bbox)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), player.color, 2)
                
                # Draw player ID and confidence
                center = player.center
                if center:
                    center_x, center_y = center
                    # Convert coordinates to integers for cv2.putText
                    text_x = int(center_x - 20)
                    text_y = int(y1 - 10)
                    cv2.putText(display_frame, f"P{player.player_id} ({player.confidence:.2f})",
                               (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, player.color, 2)
                
                # Draw keypoints if available
                if player.keypoints is not None:
                    KeypointProcessor.draw_skeleton(display_frame, player.keypoints, player.color)
                
                # Draw trajectory
                if len(player.positions) > 1:
                    for i in range(1, len(player.positions)):
                        try:
                            prev_pos = player.positions[i-1]
                            curr_pos = player.positions[i]
                            # Ensure both points are valid
                            if None not in (prev_pos, curr_pos) and len(prev_pos) == 2 and len(curr_pos) == 2:
                                cv2.line(display_frame, 
                                        (int(prev_pos[0]), int(prev_pos[1])),
                                        (int(curr_pos[0]), int(curr_pos[1])),
                                        player.color, 2)
                        except (IndexError, ValueError, TypeError) as e:
                            # Skip invalid positions
                            logger.warning(f"Error drawing trajectory: {e}")
                            continue
        
        return display_frame

# =============== Analysis Classes ===============

class SquashAnalyzer:
    """Main class for analyzing squash games"""
    def __init__(
        self,
        ball_model_path: str = "trained-models/g-ball2(white_latest).pt",
        player_model_path: str = "models/yolo11m-pose.pt",
        ball_conf_threshold: float = 0.25,
        player_conf_threshold: float = 0.35,
        max_missing_frames: int = 30,
        use_gpu: bool = True,
        use_llm: bool = True,
        llm_model_name: Optional[str] = "mistralai/Mistral-7B-Instruct-v0.2"
    ):
        """
        Initialize the analyzer
        
        Args:
            ball_model_path: Path to ball detection YOLO model
            player_model_path: Path to player detection YOLO model
            ball_conf_threshold: Confidence threshold for ball detection
            player_conf_threshold: Confidence threshold for player detection
            max_missing_frames: Maximum frames to track missing objects
            use_gpu: Whether to use GPU for model inference
            use_llm: Whether to use LLM for analysis
            llm_model_name: Hugging Face model name for LLM analysis (default: Mistral 7B Instruct)
        """
        self.ball_model_path = ball_model_path
        self.player_model_path = player_model_path
        self.ball_conf_threshold = ball_conf_threshold
        self.player_conf_threshold = player_conf_threshold
        self.max_missing_frames = max_missing_frames
        self.use_gpu = use_gpu
        self.use_llm = use_llm and HAS_TRANSFORMERS
        self.llm_model_name = llm_model_name
        
        # Initialize models
        self._initialize_models()
        
        # Initialize tracking managers
        self.player_tracking_manager = PlayerTrackingManager(
            num_players=2,
            max_missing_frames=max_missing_frames
        )
        
        # Initialize Kalman filter for ball tracking
        self.ball_kalman = KalmanFilter(process_variance=0.03, measurement_variance=0.1, disappearance_threshold=10)
        
        # Data collection
        self.ball_positions = []
        self.player_positions = []
        
    def _initialize_models(self):
        """Initialize YOLO models for ball and player detection"""
        # Initialize ball detection model
        self.ball_model = YOLO(self.ball_model_path)
        device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Check if CUDA is available for OpenCV
        if self.use_gpu and torch.cuda.is_available():
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    logger.info("CUDA is available for OpenCV")
                    # Set OpenCV DNN backend to CUDA
                    cv2.setUseOptimized(True)
                    # The following is only available if OpenCV was built with CUDA support
                    if hasattr(cv2.dnn, 'DNN_BACKEND_CUDA'):
                        logger.info("Setting OpenCV DNN backend to CUDA")
                        cv2.dnn.enableModelTypes(cv2.dnn.DNN_BACKEND_CUDA)
                        cv2.dnn.enableModelImplType(cv2.dnn.DNN_TARGET_CUDA)
            except Exception as e:
                logger.warning(f"Could not enable CUDA for OpenCV: {e}")
                logger.warning("OpenCV may not have been built with CUDA support")
        
        self.ball_model.to(device)
        
        # Initialize player detection model
        self.player_model = YOLO(self.player_model_path)
        self.player_model.to(device)
        
    def process_video(self, video_path: str, output_dir: str) -> Dict[str, str]:
        """
        Process a video and generate analysis
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save output
            
        Returns:
            Dictionary of output file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video capture
        logger.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Initialize video writers
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        ball_output_video = os.path.join(output_dir, f"{video_name}_ball_tracked.mp4")
        player_output_video = os.path.join(output_dir, f"{video_name}_player_tracked.mp4")
        
        # Try different codecs if the default doesn't work
        codecs = ['mp4v', 'avc1', 'H264', 'DIVX']
        ball_out = None
        player_out = None
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                ball_out = cv2.VideoWriter(ball_output_video, fourcc, fps, (width, height))
                player_out = cv2.VideoWriter(player_output_video, fourcc, fps, (width, height))
                
                # Test if the video writers are working
                if ball_out.isOpened() and player_out.isOpened():
                    logger.info(f"Using codec: {codec}")
                    break
                else:
                    # Close the writers and try the next codec
                    ball_out.release()
                    player_out.release()
                    ball_out = None
                    player_out = None
            except Exception as e:
                logger.warning(f"Failed to initialize video writer with codec {codec}: {e}")
                if ball_out is not None:
                    ball_out.release()
                if player_out is not None:
                    player_out.release()
                ball_out = None
                player_out = None
        
        if ball_out is None or player_out is None:
            raise RuntimeError("Could not initialize video writers with any supported codec")
        
        # Initialize CSV files
        ball_csv_path = os.path.join(output_dir, "ball_positions.csv")
        player_csv_path = os.path.join(output_dir, "player_positions.csv")
        
        ball_csv = open(ball_csv_path, 'w', newline='')
        player_csv = open(player_csv_path, 'w', newline='')
        
        ball_writer = csv.writer(ball_csv)
        player_writer = csv.writer(player_csv)
        
        # Write headers
        ball_writer.writerow(['frame', 'time_sec', 'x', 'y', 'confidence', 'estimated', 
                            'velocity_x', 'velocity_y', 'speed', 'court_region', 'court_side'])
        player_writer.writerow(['frame', 'time_sec', 'player_id', 'x', 'y', 'confidence', 
                              'keypoints', 'court_region', 'court_side'])
        
        # Process video frame by frame
        frame_number = 0
        logger.info(f"Processing video: {video_path} ({total_frames} frames)")
        
        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    time_sec = frame_number / fps
                    
                    try:
                        # Process ball tracking
                        ball_results = self.ball_model(frame, conf=self.ball_conf_threshold, verbose=False)
                        ball_frame = self._process_ball_tracking(frame, ball_results, frame_number, time_sec, ball_writer)
                        
                        # Process player tracking
                        player_results = self.player_model.track(frame, conf=self.player_conf_threshold, 
                                                            persist=True, verbose=False, classes=0)
                        player_frame = self._process_player_tracking(frame, player_results, frame_number, time_sec, player_writer)
                        
                        # Write frames
                        ball_out.write(ball_frame)
                        player_out.write(player_frame)
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_number}: {e}")
                        traceback.print_exc()
                        # Continue with next frame
                    
                    frame_number += 1
                    pbar.update(1)
        finally:
            # Release resources
            cap.release()
            ball_out.release()
            player_out.release()
            ball_csv.close()
            player_csv.close()
            
        logger.info(f"Processed {frame_number} frames")
        
        # Generate analysis
        analysis_results = self._generate_analysis(ball_csv_path, player_csv_path, output_dir)
        
        return {
            'ball_video': ball_output_video,
            'player_video': player_output_video,
            'ball_csv': ball_csv_path,
            'player_csv': player_csv_path,
            'analysis': analysis_results
        }
    
    def _process_ball_tracking(self, frame: np.ndarray, results: Any, frame_number: int, 
                             time_sec: float, csv_writer: csv.writer) -> np.ndarray:
        """Process ball tracking for a single frame"""
        display_frame = frame.copy()
        
        try:
            # Define variables to track ball confidence and coordinates
            ball_confidence = 0.0
            center_x, center_y = None, None
            estimated = True  # Flag to indicate if position is estimated or directly detected
            ball_detected = False
            detection_confidence = 0.0
            
            # Process ball detection
            if results and len(results[0].boxes) > 0:
                # Sort boxes by confidence
                boxes = results[0].boxes
                confidences = [float(box.conf[0].cpu().numpy()) for box in boxes]
                sorted_indices = np.argsort(confidences)[::-1]  # Highest confidence first
                
                # Take the highest confidence detection (likely to be the ball)
                box_idx = sorted_indices[0]
                box = boxes[box_idx]
                detection_confidence = float(box.conf[0].cpu().numpy())
                
                # Additional checks to verify this is likely the ball
                if detection_confidence >= self.ball_conf_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Calculate width and height of detection
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Verify ball-like aspect ratio (should be roughly square)
                    aspect_ratio = width / max(height, 1e-5)  # Avoid division by zero
                    
                    if 0.7 <= aspect_ratio <= 1.3 and max(width, height) < 100:  # Ball should be relatively small and square
                        # Get center position
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        ball_confidence = detection_confidence
                        ball_detected = True
                        estimated = False
                        
                        # Draw actual detection
                        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Apply Kalman filter with detection confidence
            if ball_detected:
                # We have a detection, update Kalman with measurement
                filtered_pos, confidence = self.ball_kalman.update(np.array([center_x, center_y]), ball_confidence)
            else:
                # No detection, use Kalman prediction
                filtered_pos, confidence = self.ball_kalman.update(None)
                if confidence > 0:
                    # If we have a valid prediction, use it
                    center_x, center_y = filtered_pos
                    ball_confidence = confidence
                    estimated = True
                else:
                    # No valid prediction, bail out
                    center_x, center_y = None, None
            
            # If we have a valid position (detected or predicted), draw and save it
            if center_x is not None and center_y is not None:
                # Check if it's within frame bounds
                if 0 <= center_x < frame.shape[1] and 0 <= center_y < frame.shape[0]:
                    velocity = self.ball_kalman.get_velocity()
                    
                    # Calculate speed
                    speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                    
                    # Determine court region and side
                    court_region = self._get_court_region(center_y, frame.shape[0])
                    court_side = self._get_court_side(center_x, frame.shape[1])
                    
                    # Save to CSV
                    csv_writer.writerow([
                        frame_number, time_sec, center_x, center_y, ball_confidence, estimated,
                        velocity[0], velocity[1], speed, court_region, court_side
                    ])
                    
                    # Draw ball position
                    if estimated:
                        # Use different color for estimated positions
                        cv2.circle(display_frame, (int(center_x), int(center_y)), 5, (0, 165, 255), -1)
                        cv2.putText(display_frame, f"Ball (est: {ball_confidence:.2f})", 
                                  (int(center_x) + 10, int(center_y) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    else:
                        # Use green for actual detections
                        cv2.circle(display_frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                        cv2.putText(display_frame, f"Ball ({ball_confidence:.2f})", 
                                  (int(center_x) + 10, int(center_y) - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Draw velocity vector to show ball direction
                    end_x = int(center_x + velocity[0] * 3)
                    end_y = int(center_y + velocity[1] * 3)
                    cv2.arrowedLine(display_frame, (int(center_x), int(center_y)), 
                                  (end_x, end_y), (0, 0, 255), 2)
                
                # Reset Kalman filter if tracking is likely lost
                if ball_confidence < 0.1:
                    self.ball_kalman.reset_if_lost()
        except Exception as e:
            logger.error(f"Error in ball tracking: {e}")
            traceback.print_exc()
        
        return display_frame
    
    def _process_player_tracking(self, frame: np.ndarray, results: Any, frame_number: int,
                               time_sec: float, csv_writer: csv.writer) -> np.ndarray:
        """Process player tracking for a single frame"""
        display_frame = frame.copy()
        
        try:
            # Process player detections
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                keypoints = results[0].keypoints
                
                # Update tracking manager
                self.player_tracking_manager.update(results[0], frame)
                
                # Draw tracking visualization
                display_frame = self.player_tracking_manager.visualize_tracking(display_frame)
                
                # Save player data to CSV
                for player_id, player in self.player_tracking_manager.players.items():
                    if player.bbox is not None:
                        center = player.center
                        if center:
                            center_x, center_y = center
                            court_region = self._get_court_region(center_y, frame.shape[0])
                            court_side = self._get_court_side(center_x, frame.shape[1])
                            
                            # Convert keypoints properly for JSON serialization
                            keypoints_data = []
                            if player.keypoints is not None:
                                try:
                                    # If already a numpy array, just convert to list
                                    if isinstance(player.keypoints, np.ndarray):
                                        keypoints_data = player.keypoints.tolist()
                                    # If it's a list, use it directly
                                    elif isinstance(player.keypoints, list):
                                        keypoints_data = player.keypoints
                                    # If it's an Ultralytics Keypoints object
                                    elif hasattr(player.keypoints, 'xy') and hasattr(player.keypoints, 'conf'):
                                        # Get coordinates and confidence values
                                        kpts_xy = player.keypoints.xy.cpu().numpy()
                                        conf = player.keypoints.conf.cpu().numpy()
                                        
                                        # Fix shape issues if needed
                                        num_keypoints = kpts_xy.shape[0]
                                        if len(kpts_xy.shape) > 2:  # If shape is (1, 17, 2) or similar
                                            kpts_xy = kpts_xy.reshape(num_keypoints, 2)
                                        
                                        if len(conf.shape) > 1:  # If shape is (1, 17) or similar
                                            conf = conf.reshape(num_keypoints)
                                        
                                        # Create a list of [x, y, conf] for each keypoint
                                        keypoints_data = []
                                        for j in range(num_keypoints):
                                            keypoints_data.append([
                                                float(kpts_xy[j, 0]),
                                                float(kpts_xy[j, 1]),
                                                float(conf[j])
                                            ])
                                except Exception as e:
                                    logger.warning(f"Error serializing keypoints: {e}")
                                    # Log detailed information to help debug
                                    logger.warning(f"Keypoint type: {type(player.keypoints)}")
                                    if hasattr(player.keypoints, '__dict__'):
                                        logger.warning(f"Keypoint attributes: {player.keypoints.__dict__}")
                                    else:
                                        logger.warning(f"Available attributes: {dir(player.keypoints)}")
                                    # Fall back to empty list
                                    keypoints_data = []
                            
                            csv_writer.writerow([
                                frame_number, time_sec, player_id,
                                center_x, center_y, player.confidence,
                                json.dumps(keypoints_data),
                                court_region, court_side
                            ])
        except Exception as e:
            logger.error(f"Error in player tracking: {e}")
            traceback.print_exc()
        
        return display_frame
    
    def _get_court_region(self, y: float, height: int) -> str:
        """Determine court region based on y position"""
        if y < height * 0.33:
            return "Front"
        elif y < height * 0.66:
            return "Middle"
        else:
            return "Back"
    
    def _get_court_side(self, x: float, width: int) -> str:
        """Determine court side based on x position"""
        return "Left" if x < width / 2 else "Right"
    
    def _generate_analysis(self, ball_csv_path: str, player_csv_path: str, output_dir: str) -> Dict[str, str]:
        """Generate analysis of the game"""
        # Load data
        ball_df = pd.read_csv(ball_csv_path)
        player_df = pd.read_csv(player_csv_path)
        
        # Generate visualizations
        viz_paths = self._generate_visualizations(ball_df, player_df, output_dir)
        
        # Generate LLM analysis if enabled
        llm_analysis = None
        if self.use_llm and self.llm_model_name:
            llm_analysis = self._generate_llm_analysis(ball_df, player_df, output_dir)
        
        return {
            'visualizations': viz_paths,
            'llm_analysis': llm_analysis
        }
    
    def _generate_visualizations(self, ball_df: pd.DataFrame, player_df: pd.DataFrame, 
                               output_dir: str) -> Dict[str, str]:
        """Generate analysis visualizations"""
        viz_paths = {}
        
        # Calculate FPS from time_sec differences for later use
        time_diffs = ball_df['time_sec'].diff().dropna()
        if len(time_diffs) > 0:
            avg_frame_time = time_diffs.mean()
            fps = 1 / avg_frame_time if avg_frame_time > 0 else 30  # Default to 30 if can't calculate
        else:
            fps = 30  # Default
        
        # Ball trajectory heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pd.crosstab(
                pd.cut(ball_df['y'], bins=20),
                pd.cut(ball_df['x'], bins=20)
            ),
            cmap='hot'
        )
        plt.title('Ball Position Heatmap')
        viz_paths['ball_heatmap'] = os.path.join(output_dir, 'ball_heatmap.png')
        plt.savefig(viz_paths['ball_heatmap'])
        plt.close()
        
        # Player movement heatmap
        plt.figure(figsize=(10, 8))
        for player_id in player_df['player_id'].unique():
            player_data = player_df[player_df['player_id'] == player_id]
            plt.scatter(player_data['x'], player_data['y'], 
                       alpha=0.5, label=f'Player {player_id}')
        plt.title('Player Court Coverage')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        viz_paths['player_coverage'] = os.path.join(output_dir, 'player_coverage.png')
        plt.savefig(viz_paths['player_coverage'])
        plt.close()
        
        # Ball speed over time
        plt.figure(figsize=(12, 6))
        smoothed_speed = savgol_filter(ball_df['speed'], 
                                      min(51, len(ball_df) - len(ball_df) % 2 - 1), 3)
        plt.plot(ball_df['time_sec'], smoothed_speed)
        plt.title('Ball Speed Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speed (pixels/frame)')
        plt.grid(True, alpha=0.3)
        viz_paths['ball_speed'] = os.path.join(output_dir, 'ball_speed.png')
        plt.savefig(viz_paths['ball_speed'])
        plt.close()
        
        # Player distance from center over time
        plt.figure(figsize=(12, 6))
        # Calculate court center
        court_center_x = player_df['x'].mean()
        court_center_y = player_df['y'].mean()
        
        for player_id in player_df['player_id'].unique():
            player_data = player_df[player_df['player_id'] == player_id]
            # Calculate distance from center
            player_data['center_distance'] = np.sqrt(
                (player_data['x'] - court_center_x)**2 + 
                (player_data['y'] - court_center_y)**2
            )
            # Smooth the distance
            if len(player_data) > 10:
                smoothed_distance = savgol_filter(
                    player_data['center_distance'],
                    min(51, len(player_data) - len(player_data) % 2 - 1), 3
                )
                plt.plot(player_data['time_sec'], smoothed_distance, 
                       label=f'Player {player_id}')
        
        plt.title('Player Distance from Court Center')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Distance (pixels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        viz_paths['center_distance'] = os.path.join(output_dir, 'center_distance.png')
        plt.savefig(viz_paths['center_distance'])
        plt.close()
        
        # Player heatmaps (one for each player)
        for player_id in player_df['player_id'].unique():
            player_data = player_df[player_df['player_id'] == player_id]
            plt.figure(figsize=(10, 8))
            sns.kdeplot(
                x=player_data['x'],
                y=player_data['y'],
                cmap='viridis',
                fill=True,
                bw_adjust=0.7
            )
            plt.title(f'Player {player_id} Court Coverage Heatmap')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            player_heatmap_path = os.path.join(output_dir, f'player{player_id}_heatmap.png')
            viz_paths[f'player{player_id}_heatmap'] = player_heatmap_path
            plt.savefig(player_heatmap_path)
            plt.close()
            
        # Court region distribution pie charts
        for player_id in player_df['player_id'].unique():
            player_data = player_df[player_df['player_id'] == player_id]
            plt.figure(figsize=(8, 8))
            region_counts = player_data['court_region'].value_counts()
            plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%')
            plt.title(f'Player {player_id} Court Region Distribution')
            region_chart_path = os.path.join(output_dir, f'player{player_id}_regions.png')
            viz_paths[f'player{player_id}_regions'] = region_chart_path
            plt.savefig(region_chart_path)
            plt.close()
            
        # Side preference distribution pie charts
        for player_id in player_df['player_id'].unique():
            player_data = player_df[player_df['player_id'] == player_id]
            plt.figure(figsize=(8, 8))
            side_counts = player_data['court_side'].value_counts()
            plt.pie(side_counts, labels=side_counts.index, autopct='%1.1f%%')
            plt.title(f'Player {player_id} Court Side Preference')
            side_chart_path = os.path.join(output_dir, f'player{player_id}_sides.png')
            viz_paths[f'player{player_id}_sides'] = side_chart_path
            plt.savefig(side_chart_path)
            plt.close()
            
        # Generate summary statistics table
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        
        # Create summary statistics
        summary_data = []
        
        # Game duration
        duration = ball_df['time_sec'].max()
        summary_data.append(["Game Duration", f"{duration:.2f} seconds"])
        
        # Ball statistics
        avg_speed = ball_df['speed'].mean()
        max_speed = ball_df['speed'].max()
        ball_front = (ball_df['court_region'] == 'Front').mean() * 100
        ball_middle = (ball_df['court_region'] == 'Middle').mean() * 100
        ball_back = (ball_df['court_region'] == 'Back').mean() * 100
        ball_left = (ball_df['court_side'] == 'Left').mean() * 100
        ball_right = (ball_df['court_side'] == 'Right').mean() * 100
        
        summary_data.extend([
            ["Average Ball Speed", f"{avg_speed:.2f} pixels/frame"],
            ["Maximum Ball Speed", f"{max_speed:.2f} pixels/frame"],
            ["Ball in Front Court", f"{ball_front:.1f}%"],
            ["Ball in Middle Court", f"{ball_middle:.1f}%"],
            ["Ball in Back Court", f"{ball_back:.1f}%"],
            ["Ball on Left Side", f"{ball_left:.1f}%"],
            ["Ball on Right Side", f"{ball_right:.1f}%"],
        ])
        
        # Player statistics
        for player_id in player_df['player_id'].unique():
            player_data = player_df[player_df['player_id'] == player_id]
            
            # Calculate player movement
            movement = 0
            for i in range(1, len(player_data)):
                if i > 0 and player_data.iloc[i-1]['frame'] + 1 == player_data.iloc[i]['frame']:
                    # Continuous frames, calculate distance moved
                    x1, y1 = player_data.iloc[i-1]['x'], player_data.iloc[i-1]['y']
                    x2, y2 = player_data.iloc[i]['x'], player_data.iloc[i]['y']
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    movement += distance
            
            # Court position percentages
            front_pct = (player_data['court_region'] == 'Front').mean() * 100
            middle_pct = (player_data['court_region'] == 'Middle').mean() * 100
            back_pct = (player_data['court_region'] == 'Back').mean() * 100
            left_pct = (player_data['court_side'] == 'Left').mean() * 100
            right_pct = (player_data['court_side'] == 'Right').mean() * 100
            
            summary_data.extend([
                [f"Player {player_id} Total Movement", f"{movement:.2f} pixels"],
                [f"Player {player_id} in Front Court", f"{front_pct:.1f}%"],
                [f"Player {player_id} in Middle Court", f"{middle_pct:.1f}%"],
                [f"Player {player_id} in Back Court", f"{back_pct:.1f}%"],
                [f"Player {player_id} on Left Side", f"{left_pct:.1f}%"],
                [f"Player {player_id} on Right Side", f"{right_pct:.1f}%"],
            ])
            
        # Calculate advanced metrics
        for player_id in player_df['player_id'].unique():
            player_data = player_df[player_df['player_id'] == player_id]
            
            # Calculate reaction time (average time to move toward ball after it changes direction)
            # This is a simplified approximation
            reaction_times = []
            for i in range(1, min(len(ball_df), len(player_data))):
                if i > 0 and i < len(ball_df)-1:
                    # Check if ball changed direction significantly
                    vx1 = ball_df.iloc[i-1]['velocity_x']
                    vx2 = ball_df.iloc[i]['velocity_x']
                    vy1 = ball_df.iloc[i-1]['velocity_y']
                    vy2 = ball_df.iloc[i]['velocity_y']
                    
                    # Calculate angle change
                    angle1 = np.arctan2(vy1, vx1)
                    angle2 = np.arctan2(vy2, vx2)
                    angle_change = abs(angle2 - angle1)
                    
                    # If angle changed significantly, check player's reaction
                    if angle_change > 0.5:  # threshold in radians (about 30 degrees)
                        # Look at next few frames for player movement toward ball
                        ball_pos = (ball_df.iloc[i]['x'], ball_df.iloc[i]['y'])
                        player_distances = []
                        
                        # Look at player positions in subsequent frames
                        for j in range(i, min(i+15, len(player_data))):
                            player_pos = (player_data.iloc[j]['x'], player_data.iloc[j]['y'])
                            dist = np.sqrt((player_pos[0]-ball_pos[0])**2 + (player_pos[1]-ball_pos[1])**2)
                            player_distances.append(dist)
                        
                        # Check if player moved toward the ball
                        if len(player_distances) > 5 and player_distances[0] > player_distances[-1]:
                            # Find first frame where player starts moving toward ball
                            for j in range(1, len(player_distances)):
                                if player_distances[j] < player_distances[j-1]:
                                    reaction_times.append(j / fps)  # Convert frames to seconds
                                    break
            
            # Calculate average reaction time if we have data
            if reaction_times:
                avg_reaction = sum(reaction_times) / len(reaction_times)
                summary_data.append([f"Player {player_id} Avg Reaction Time", f"{avg_reaction:.2f} seconds"])
        
        # Calculate shot stats
        shot_count = 0
        rally_lengths = []
        current_rally = 0
        last_direction = None
        
        # Detect shots based on significant changes in ball velocity
        for i in range(1, len(ball_df)):
            if i > 0:
                vx1 = ball_df.iloc[i-1]['velocity_x']
                vx2 = ball_df.iloc[i]['velocity_x']
                vy1 = ball_df.iloc[i-1]['velocity_y']
                vy2 = ball_df.iloc[i]['velocity_y']
                
                # Calculate acceleration magnitude
                acc_x = abs(vx2 - vx1)
                acc_y = abs(vy2 - vy1)
                acc_mag = np.sqrt(acc_x**2 + acc_y**2)
                
                # If significant acceleration, count as potential shot
                if acc_mag > 20:  # Adjust threshold as needed
                    # Check if direction changed
                    dir1 = np.sign(vx1) + np.sign(vy1) * 2  # Simple direction encoding
                    dir2 = np.sign(vx2) + np.sign(vy2) * 2
                    
                    if dir1 != dir2:
                        shot_count += 1
                        current_rally += 1
                        last_direction = dir2
                
                # Check for rally end (ball stops moving)
                if np.sqrt(vx2**2 + vy2**2) < 0.5 and current_rally > 0:
                    if current_rally > 1:  # Only count rallies with more than one shot
                        rally_lengths.append(current_rally)
                    current_rally = 0
        
        # Add shot statistics to summary
        summary_data.extend([
            ["Estimated Shot Count", str(shot_count)],
            ["Average Rally Length", f"{np.mean(rally_lengths):.1f} shots" if rally_lengths else "N/A"],
            ["Longest Rally", f"{max(rally_lengths)} shots" if rally_lengths else "N/A"],
        ])
        
        # Create a table with the summary statistics
        table = plt.table(
            cellText=summary_data,
            colWidths=[0.3, 0.7],
            loc='center',
            cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            cell.set_edgecolor('lightgrey')
            
        plt.title('Game Summary Statistics', fontsize=16, pad=20)
        summary_path = os.path.join(output_dir, 'game_summary.png')
        viz_paths['game_summary'] = summary_path
        plt.savefig(summary_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Save summary statistics as CSV
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
        summary_csv = os.path.join(output_dir, 'game_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        viz_paths['summary_csv'] = summary_csv
        
        return viz_paths
    
    def _generate_llm_analysis(self, ball_df: pd.DataFrame, player_df: pd.DataFrame, 
                             output_dir: str) -> Optional[str]:
        """Generate analysis using LLM"""
        if not self.use_llm or not self.llm_model_name:
            return None
        
        try:
            # Calculate FPS from time_sec differences
            time_diffs = ball_df['time_sec'].diff().dropna()
            if len(time_diffs) > 0:
                avg_frame_time = time_diffs.mean()
                fps = 1 / avg_frame_time if avg_frame_time > 0 else 30  # Default to 30 if can't calculate
            else:
                fps = 30  # Default
            
            # Prepare analysis data with more detailed statistics
            
            # Shot detection (simplified)
            shot_frames = []
            for i in range(2, len(ball_df)):
                vx1 = ball_df.iloc[i-1]['velocity_x']
                vx2 = ball_df.iloc[i]['velocity_x']
                vy1 = ball_df.iloc[i-1]['velocity_y']
                vy2 = ball_df.iloc[i]['velocity_y']
                
                # Calculate acceleration magnitude
                acc_x = abs(vx2 - vx1)
                acc_y = abs(vy2 - vy1)
                acc_mag = np.sqrt(acc_x**2 + acc_y**2)
                
                # If significant acceleration, mark as potential shot
                if acc_mag > 20:  # Threshold for shot detection
                    shot_frames.append(i)
            
            # Detect rallies
            rallies = []
            current_rally = []
            for i in range(len(shot_frames)):
                if i == 0 or shot_frames[i] - shot_frames[i-1] < 30:  # Frames threshold for same rally
                    current_rally.append(shot_frames[i])
                else:
                    if len(current_rally) > 1:  # Only count rallies with more than one shot
                        rallies.append(current_rally)
                    current_rally = [shot_frames[i]]
            
            if len(current_rally) > 1:
                rallies.append(current_rally)
            
            # Calculate rally statistics
            rally_lengths = [len(rally) for rally in rallies]
            rally_durations = []
            for rally in rallies:
                if rally:
                    start_time = ball_df.iloc[rally[0]]['time_sec']
                    end_time = ball_df.iloc[rally[-1]]['time_sec']
                    rally_durations.append(end_time - start_time)
            
            # Calculate movement patterns
            player_movements = {}
            for player_id in player_df['player_id'].unique():
                player_data = player_df[player_df['player_id'] == player_id]
                
                # Calculate total distance moved
                total_distance = 0
                distances_per_sec = []
                for i in range(1, len(player_data)):
                    if player_data.iloc[i-1]['frame'] + 1 == player_data.iloc[i]['frame']:
                        # Continuous frames
                        x1, y1 = player_data.iloc[i-1]['x'], player_data.iloc[i-1]['y']
                        x2, y2 = player_data.iloc[i]['x'], player_data.iloc[i]['y']
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        total_distance += distance
                        
                        # Group by second for movement rate
                        time_sec = int(player_data.iloc[i]['time_sec'])
                        if time_sec < len(distances_per_sec):
                            distances_per_sec[time_sec] += distance
                        else:
                            # Extend list if needed
                            distances_per_sec.extend([0] * (time_sec - len(distances_per_sec) + 1))
                            distances_per_sec[time_sec] = distance
                
                # Calculate court coverage statistics
                front_time = (player_data['court_region'] == 'Front').mean()
                middle_time = (player_data['court_region'] == 'Middle').mean()
                back_time = (player_data['court_region'] == 'Back').mean()
                left_time = (player_data['court_side'] == 'Left').mean()
                right_time = (player_data['court_side'] == 'Right').mean()
                
                # Calculate movement rates
                avg_movement_rate = np.mean(distances_per_sec) if distances_per_sec else 0
                max_movement_rate = np.max(distances_per_sec) if distances_per_sec else 0
                
                player_movements[str(player_id)] = {
                    'total_distance': total_distance,
                    'avg_movement_rate': avg_movement_rate,
                    'max_movement_rate': max_movement_rate,
                    'front_time_pct': front_time * 100,
                    'middle_time_pct': middle_time * 100,
                    'back_time_pct': back_time * 100,
                    'left_time_pct': left_time * 100,
                    'right_time_pct': right_time * 100
                }
            
            # Calculate ball statistics
            ball_front_time = (ball_df['court_region'] == 'Front').mean() * 100
            ball_middle_time = (ball_df['court_region'] == 'Middle').mean() * 100
            ball_back_time = (ball_df['court_region'] == 'Back').mean() * 100
            ball_left_time = (ball_df['court_side'] == 'Left').mean() * 100
            ball_right_time = (ball_df['court_side'] == 'Right').mean() * 100
            
            # Calculate average ball speeds in different court regions
            front_speed = ball_df[ball_df['court_region'] == 'Front']['speed'].mean()
            middle_speed = ball_df[ball_df['court_region'] == 'Middle']['speed'].mean()
            back_speed = ball_df[ball_df['court_region'] == 'Back']['speed'].mean()
            
            # Player positioning relative to ball
            player_ball_distances = {}
            for player_id in player_df['player_id'].unique():
                player_data = player_df[player_df['player_id'] == player_id]
                
                # Calculate average distance to ball
                distances = []
                for i in range(min(len(player_data), len(ball_df))):
                    if player_data.iloc[i]['frame'] == ball_df.iloc[i]['frame']:
                        px, py = player_data.iloc[i]['x'], player_data.iloc[i]['y']
                        bx, by = ball_df.iloc[i]['x'], ball_df.iloc[i]['y']
                        distance = np.sqrt((px-bx)**2 + (py-by)**2)
                        distances.append(distance)
                
                avg_distance = np.mean(distances) if distances else 0
                
                player_ball_distances[str(player_id)] = {
                    'avg_distance_to_ball': avg_distance
                }
            
            # Enhanced analysis data
            analysis_data = {
                'game_stats': {
                    'duration_sec': ball_df['time_sec'].max(),
                    'total_frames': len(ball_df),
                    'fps': fps,
                    'shot_count': len(shot_frames),
                    'rally_count': len(rallies),
                    'avg_rally_length': np.mean(rally_lengths) if rally_lengths else 0,
                    'max_rally_length': np.max(rally_lengths) if rally_lengths else 0,
                    'avg_rally_duration': np.mean(rally_durations) if rally_durations else 0,
                    'max_rally_duration': np.max(rally_durations) if rally_durations else 0,
                },
                'ball_stats': {
                    'mean_position': (ball_df['x'].mean(), ball_df['y'].mean()),
                    'position_std': (ball_df['x'].std(), ball_df['y'].std()),
                    'court_region_pct': {
                        'front': ball_front_time,
                        'middle': ball_middle_time,
                        'back': ball_back_time
                    },
                    'court_side_pct': {
                        'left': ball_left_time,
                        'right': ball_right_time
                    },
                    'speeds_by_region': {
                        'front': front_speed,
                        'middle': middle_speed,
                        'back': back_speed
                    },
                    'avg_speed': ball_df['speed'].mean(),
                    'max_speed': ball_df['speed'].max(),
                    'speed_percentiles': {
                        '25th': ball_df['speed'].quantile(0.25),
                        '50th': ball_df['speed'].quantile(0.50),
                        '75th': ball_df['speed'].quantile(0.75),
                        '90th': ball_df['speed'].quantile(0.90)
                    }
                },
                'player_stats': {
                    str(player_id): {
                        'mean_position': (group['x'].mean(), group['y'].mean()),
                        'position_std': (group['x'].std(), group['y'].std()),
                        'court_coverage': {
                            'front': player_movements[str(player_id)]['front_time_pct'],
                            'middle': player_movements[str(player_id)]['middle_time_pct'],
                            'back': player_movements[str(player_id)]['back_time_pct'],
                            'left': player_movements[str(player_id)]['left_time_pct'],
                            'right': player_movements[str(player_id)]['right_time_pct']
                        },
                        'movement': {
                            'total_distance': player_movements[str(player_id)]['total_distance'],
                            'avg_movement_rate': player_movements[str(player_id)]['avg_movement_rate'],
                            'max_movement_rate': player_movements[str(player_id)]['max_movement_rate']
                        },
                        'ball_interaction': {
                            'avg_distance_to_ball': player_ball_distances[str(player_id)]['avg_distance_to_ball']
                        }
                    }
                    for player_id, group in player_df.groupby('player_id')
                }
            }
            
            # Generate prompt with enhanced data
            prompt = f"""
            You are an expert squash coach and analyst with 20+ years of experience coaching professional players. 
            Analyze the following squash game data and provide specialized feedback:
            
            {json.dumps(analysis_data, indent=2, cls=NumpyEncoder)}
            
            Please provide:
            1. An overall assessment of the game pattern and quality level (beginner/intermediate/advanced)
            2. Detailed breakdown of each player's playing style, movement patterns, and court positioning
            3. Specific strengths for each player, backed by data (e.g., "Player 1 covers the front court well, spending {analysis_data['player_stats']['1']['court_coverage']['front']:.1f}% of time there")
            4. Specific weaknesses for each player with clear examples from the data
            5. Tactical analysis of rallies and shot selection patterns
            6. Comparison between players: who was more dominant and why?
            7. Three specific training recommendations for each player based on their stats
            8. One game strategy recommendation for each player for their next match against this opponent
            
            Your analysis should be highly technical, data-driven, and specific - as if you were presenting 
            a professional analysis to competitive players. Reference specific statistics from the data to 
            support all your observations and recommendations.
            """
            
            # Initialize model and generate analysis
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                device_map="auto",
                load_in_4bit=True
            )
            
            text_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            result = text_generator(prompt, return_full_text=False)
            analysis = result[0]["generated_text"]
            
            # Save analysis
            analysis_path = os.path.join(output_dir, 'coach_analysis.txt')
            with open(analysis_path, 'w') as f:
                f.write(analysis)
            
            return analysis_path
            
        except Exception as e:
            logger.error(f"Error generating LLM analysis: {e}")
            return None

# =============== Keypoint Processing ===============

class KeypointProcessor:
    """Process and visualize keypoints for player pose estimation"""
    
    # COCO keypoint pairs for skeleton drawing
    SKELETON = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
        [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
    ]
    
    @staticmethod
    def draw_skeleton(frame: np.ndarray, keypoints: Any, color: Tuple[int, int, int]) -> None:
        """
        Draw skeleton on frame using keypoints
        
        Args:
            frame: Frame to draw on
            keypoints: Array/list/tensor of keypoints [x, y, conf]
            color: Color for drawing
        """
        try:
            # Check keypoints shape/format
            if keypoints is None:
                return
                
            # Convert keypoints to numpy array if it's not already
            kpts_array = None
            conf_array = None
            
            # Handle Ultralytics Keypoints class specifically
            if hasattr(keypoints, 'xy') and hasattr(keypoints, 'conf'):
                # This is an Ultralytics Keypoints object
                kpts_xy = keypoints.xy.cpu().numpy()
                conf = keypoints.conf.cpu().numpy()
                
                # Debug the shapes
                logger.info(f"Keypoints xy shape: {kpts_xy.shape}")
                logger.info(f"Keypoints conf shape: {conf.shape}")
                
                # Create array with [x, y, conf] format
                # Determine number of keypoints carefully
                if len(kpts_xy.shape) == 3:  # shape like (1, 17, 2)
                    num_keypoints = kpts_xy.shape[1]
                    kpts_xy = kpts_xy.reshape(num_keypoints, 2)
                elif len(kpts_xy.shape) == 2:  # shape like (17, 2)
                    num_keypoints = kpts_xy.shape[0]
                else:
                    # Handle unexpected shape
                    logger.warning(f"Unexpected keypoints xy shape: {kpts_xy.shape}")
                    # Try to infer from conf shape if possible
                    if hasattr(conf, 'shape') and len(conf.shape) > 0:
                        if len(conf.shape) == 1:  # shape like (17,)
                            num_keypoints = conf.shape[0]
                        elif len(conf.shape) == 2:  # shape like (1, 17)
                            num_keypoints = conf.shape[1]
                        else:
                            # Last resort
                            num_keypoints = kpts_xy.size // 2
                    else:
                        # Last resort
                        num_keypoints = kpts_xy.size // 2
                
                kpts_array = np.zeros((num_keypoints, 3))
                
                # Ensure kpts_xy has the right shape for assignment
                if kpts_xy.size == num_keypoints * 2:
                    # If flattened or wrong shape, reshape correctly
                    if kpts_xy.shape != (num_keypoints, 2):
                        try:
                            kpts_xy = kpts_xy.reshape(num_keypoints, 2)
                        except Exception as e:
                            logger.error(f"Failed to reshape keypoints xy: {e}")
                            logger.error(f"kpts_xy shape: {kpts_xy.shape}, size: {kpts_xy.size}, num_keypoints: {num_keypoints}")
                            # Create a proper shaped array by manually copying values
                            flat_kpts = kpts_xy.flatten()
                            new_kpts_xy = np.zeros((num_keypoints, 2))
                            for j in range(min(num_keypoints, len(flat_kpts) // 2)):
                                new_kpts_xy[j, 0] = flat_kpts[j*2]
                                new_kpts_xy[j, 1] = flat_kpts[j*2+1]
                            kpts_xy = new_kpts_xy

                    # Assign xy coordinates safely
                    try:
                        kpts_array[:, 0:2] = kpts_xy
                    except Exception as e:
                        logger.error(f"Failed to assign keypoints xy to output array: {e}")
                        # Fallback: copy element by element
                        for j in range(min(num_keypoints, kpts_xy.shape[0])):
                            if j < kpts_xy.shape[0] and kpts_xy.shape[1] >= 2:
                                kpts_array[j, 0] = kpts_xy[j, 0]
                                kpts_array[j, 1] = kpts_xy[j, 1]
                        
                    # Handle conf shape similarly to xy
                    if hasattr(conf, 'shape'):
                        # Handle 2D array
                        if len(conf.shape) == 2:  # shape like (1, 17)
                            try:
                                conf = conf.reshape(num_keypoints)
                            except Exception as e:
                                logger.error(f"Failed to reshape confidence: {e}")
                                # Create properly shaped confidence
                                flat_conf = conf.flatten()
                                conf = np.zeros(num_keypoints)
                                for j in range(min(num_keypoints, len(flat_conf))):
                                    conf[j] = flat_conf[j]
                        # Handle 1D array with wrong length
                        elif len(conf.shape) == 1 and conf.shape[0] != num_keypoints:
                            # If lengths don't match, resize
                            new_conf = np.zeros(num_keypoints)
                            for j in range(min(num_keypoints, conf.shape[0])):
                                new_conf[j] = conf[j]
                            conf = new_conf
                        else:
                            # If conf has no shape attribute, create default conf
                            conf = np.ones(num_keypoints) * 0.5
                        
                        # Assign confidence safely
                        try:
                            kpts_array[:, 2] = conf
                        except Exception as e:
                            logger.error(f"Failed to assign confidence to output array: {e}")
                            # Fallback: copy element by element
                            for j in range(min(num_keypoints, len(conf))):
                                if j < len(conf):
                                    kpts_array[j, 2] = conf[j]
                    # If it's a numpy array already
                    elif isinstance(keypoints, np.ndarray):
                        kpts_array = keypoints
                    # If it's a list
                    elif isinstance(keypoints, list):
                        kpts_array = np.array(keypoints)
                    # If it's a tensor or other object
                    else:
                        # Try to convert to numpy array through different paths
                        try:
                            if hasattr(keypoints, 'cpu'):
                                cpu_keypoints = keypoints.cpu()
                                if hasattr(cpu_keypoints, 'numpy'):
                                    kpts_array = cpu_keypoints.numpy()
                                elif hasattr(cpu_keypoints, 'detach'):
                                    kpts_array = cpu_keypoints.detach().numpy()
                            elif hasattr(keypoints, 'numpy'):
                                kpts_array = keypoints.numpy()
                        except Exception as e:
                            logger.warning(f"Failed to convert keypoints to numpy array: {e}")
                            return
                    
                    # If conversion failed or keypoints array is empty
                    if kpts_array is None or len(kpts_array) == 0:
                        logger.warning("Empty keypoints array or conversion failed")
                        return
                        
                    # Draw keypoints
                    for i, kp in enumerate(kpts_array):
                        # Check if keypoint has enough elements for x, y, confidence
                        if len(kp) < 3:
                            continue
                        
                        # Check confidence threshold
                        conf = float(kp[2])  # Ensure it's a float
                        if conf > 0.5:  # Only draw if confidence > 0.5
                            x, y = int(float(kp[0])), int(float(kp[1]))  # Ensure they're integers
                            cv2.circle(frame, (x, y), 4, color, -1)
                    
                    # Draw skeleton
                    for pair in KeypointProcessor.SKELETON:
                        # Skip invalid indices
                        if pair[0] >= len(kpts_array) or pair[1] >= len(kpts_array):
                            continue
                        
                        pt1 = kpts_array[pair[0]]
                        pt2 = kpts_array[pair[1]]
                        
                        # Check if keypoints have enough elements and meet confidence threshold
                        if (len(pt1) >= 3 and len(pt2) >= 3 and 
                            float(pt1[2]) > 0.5 and float(pt2[2]) > 0.5):  # Only draw if both points are confident
                            x1, y1 = int(float(pt1[0])), int(float(pt1[1]))
                            x2, y2 = int(float(pt2[0])), int(float(pt2[1]))
                            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        except Exception as e:
            logger.warning(f"Error drawing skeleton: {e}")
            logger.warning(f"Keypoint type: {type(keypoints)}")
            if hasattr(keypoints, '__dict__'):
                logger.warning(f"Keypoint attributes: {keypoints.__dict__}")
            # Continue without drawing skeleton

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Analyze squash game from video')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: timestamped directory)')
    parser.add_argument('--ball-model', type=str, 
                       default="trained-models/g-ball2(white_latest).pt",
                       help='Path to ball detection model')
    parser.add_argument('--player-model', type=str,
                       default="models/yolo11m-pose.pt",
                       help='Path to player detection model')
    parser.add_argument('--ball-conf', type=float, default=0.25,
                       help='Ball detection confidence threshold')
    parser.add_argument('--player-conf', type=float, default=0.35,
                       help='Player detection confidence threshold')
    parser.add_argument('--missing-frames', type=int, default=30,
                       help='Maximum frames to track missing objects')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU inference')
    parser.add_argument('--no-llm', action='store_true',
                       help='Disable LLM-based analysis')
    parser.add_argument('--llm-model', type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                       help='Hugging Face model name for LLM analysis (default: Mistral 7B Instruct)')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("analysis_output", f"squash_analysis_{timestamp}")
    else:
        output_dir = args.output
    
    try:
        # Initialize analyzer
        analyzer = SquashAnalyzer(
            ball_model_path=args.ball_model,
            player_model_path=args.player_model,
            ball_conf_threshold=args.ball_conf,
            player_conf_threshold=args.player_conf,
            max_missing_frames=args.missing_frames,
            use_gpu=not args.cpu,
            use_llm=not args.no_llm,
            llm_model_name=args.llm_model
        )
        
        # Process video
        results = analyzer.process_video(args.video, output_dir)
        
        # Print results
        print("\nAnalysis complete!")
        print(f"Output directory: {output_dir}")
        print("\nGenerated files:")
        print(f"- Ball tracking video: {results['ball_video']}")
        print(f"- Player tracking video: {results['player_video']}")
        print(f"- Ball position data: {results['ball_csv']}")
        print(f"- Player position data: {results['player_csv']}")
        
        if results['analysis']['visualizations']:
            print("\nVisualizations:")
            for name, path in results['analysis']['visualizations'].items():
                print(f"- {name}: {path}")
        
        if results['analysis']['llm_analysis']:
            print(f"\nCoach analysis: {results['analysis']['llm_analysis']}")
        
        # Open output directory
        try:
            import platform
            import subprocess
            
            if platform.system() == "Windows":
                os.startfile(output_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
        except Exception as e:
            logger.error(f"Failed to open output directory: {e}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        traceback.print_exc()
        print("\nTips for troubleshooting:")
        print("1. Check if the video file exists and is readable")
        print("2. Verify that the model files exist in the specified paths")
        print("3. Ensure you have sufficient GPU memory if using GPU")
        print("4. Check the output directory permissions")

if __name__ == "__main__":
    main() 