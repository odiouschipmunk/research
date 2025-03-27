import cv2
import numpy as np
import torch
import os
import csv
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union, Any, Deque
from datetime import datetime
from ultralytics import YOLO
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('squash_tracker')

# Type aliases for improved readability
Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2
Vector = Tuple[float, float]  # x, y components
KeypointType = List[List[float]]  # [x, y, confidence] for each keypoint

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
    """
    Tracks a player across video frames using multiple cues:
    - Position and motion modeling
    - Appearance features
    - Pose keypoints
    - Track IDs from underlying object tracker
    
    Implements robust tracking with identity preservation and occlusion handling.
    """
    
    def __init__(
        self, 
        player_id: int, 
        max_history: int = 30, 
        appearance_history_size: int = 10,
        heatmap_resolution: Tuple[int, int] = (20, 20),
        motion_smoothing_factor: float = 0.7
    ):
        """
        Initialize player tracker
        
        Args:
            player_id: Unique identifier for the player (1 or 2)
            max_history: Maximum history of positions to keep
            appearance_history_size: Maximum history of appearance features
            heatmap_resolution: Resolution of position heatmap (width, height)
            motion_smoothing_factor: Alpha factor for motion smoothing (0-1)
        """
        self.player_id = player_id
        self.track_id: Optional[int] = None
        
        # Position and motion tracking
        self.positions: Deque[Point] = deque(maxlen=max_history)
        self.keypoints_history: Deque[KeypointType] = deque(maxlen=max_history)
        self.long_term_positions: Deque[Point] = deque(maxlen=100)
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
        
        # Visual appearance modeling
        self.appearance_history: Deque[np.ndarray] = deque(maxlen=appearance_history_size)
        
        # Set player color (Red for P1, Blue for P2)
        self.color: Tuple[int, int, int] = (0, 0, 255) if player_id == 1 else (255, 0, 0)
        
        # Position heatmap for spatial consistency checking
        self.position_heatmap: Optional[np.ndarray] = None
        self.heatmap_resolution = heatmap_resolution
        self.heatmap_width: Optional[int] = None
        self.heatmap_height: Optional[int] = None
        self.heatmap_updates: int = 0
        
    def initialize_heatmap(self, width: int, height: int) -> None:
        """
        Initialize position heatmap for the player
        
        Args:
            width: Frame width
            height: Frame height
        """
        if self.position_heatmap is None:
            self.position_heatmap = np.zeros(self.heatmap_resolution, dtype=np.float32)
            self.heatmap_width = width
            self.heatmap_height = height
    
    def update_heatmap(self, position: Point) -> None:
        """
        Update the position heatmap with current position
        
        Args:
            position: Current position (x, y)
        """
        if self.position_heatmap is None or position is None:
            return
            
        # Convert position to heatmap coordinates
        grid_x = int((position[0] / self.heatmap_width) * self.position_heatmap.shape[1])
        grid_y = int((position[1] / self.heatmap_height) * self.position_heatmap.shape[0])
        
        # Bound to valid indices
        grid_x = max(0, min(grid_x, self.position_heatmap.shape[1] - 1))
        grid_y = max(0, min(grid_y, self.position_heatmap.shape[0] - 1))
        
        # Update heatmap with exponential decay
        decay_factor = 0.95
        self.position_heatmap = self.position_heatmap * decay_factor
        self.position_heatmap[grid_y, grid_x] += 1.0
        self.heatmap_updates += 1
    
    def get_position_consistency_score(self, position: Point) -> float:
        """
        Calculate how consistent a position is with historical positions
        
        Args:
            position: Position to check (x, y)
            
        Returns:
            Consistency score between 0 and 1
        """
        if self.position_heatmap is None or self.heatmap_updates < 30 or position is None:
            return 0.5  # Neutral score if insufficient history
            
        # Convert position to heatmap coordinates
        grid_x = int((position[0] / self.heatmap_width) * self.position_heatmap.shape[1])
        grid_y = int((position[1] / self.heatmap_height) * self.position_heatmap.shape[0])
        
        # Bound to valid indices
        grid_x = max(0, min(grid_x, self.position_heatmap.shape[1] - 1))
        grid_y = max(0, min(grid_y, self.position_heatmap.shape[0] - 1))
        
        # Normalize score based on heatmap maximum
        if np.max(self.position_heatmap) > 0:
            consistency = self.position_heatmap[grid_y, grid_x] / np.max(self.position_heatmap)
            return float(consistency)
        return 0.5
    
    def update_motion_model(self, center_x: float, center_y: float) -> None:
        """
        Update velocity and acceleration estimates
        
        Args:
            center_x: Current center x position
            center_y: Current center y position
        """
        if len(self.positions) >= 2:
            prev_pos = self.positions[-1]
            prev_prev_pos = self.positions[-2]
            
            # Calculate current velocity
            curr_velocity = (center_x - prev_pos[0], center_y - prev_pos[1])
            
            # Calculate current acceleration
            prev_velocity = (prev_pos[0] - prev_prev_pos[0], prev_pos[1] - prev_prev_pos[1])
            curr_acceleration = (curr_velocity[0] - prev_velocity[0], curr_velocity[1] - prev_velocity[1])
            
            # Update velocity and acceleration estimates with smoothing
            alpha = self.motion_smoothing_factor
            self.velocity = (alpha * curr_velocity[0] + (1-alpha) * self.velocity[0],
                            alpha * curr_velocity[1] + (1-alpha) * self.velocity[1])
            self.acceleration = (alpha * curr_acceleration[0] + (1-alpha) * self.acceleration[0],
                                alpha * curr_acceleration[1] + (1-alpha) * self.acceleration[1])
        
    def update(self, detection: Detection, frame: Optional[np.ndarray] = None) -> None:
        """
        Update player state with new detection
        
        Args:
            detection: New detection data
            frame: Video frame for appearance modeling
        """
        bbox = detection.bbox
        keypoints = detection.keypoints
        confidence = detection.confidence
        
        center_x, center_y = detection.center
        height = detection.height
        
        # Initialize heatmap if needed
        if frame is not None and self.position_heatmap is None:
            self.initialize_heatmap(frame.shape[1], frame.shape[0])
        
        # Update motion model
        self.update_motion_model(center_x, center_y)
        
        # Update appearance descriptor if frame provided
        if frame is not None:
            roi = detection.extract_roi(frame)
            if roi is not None and roi.size > 0:
                # Extract appearance features
                hist_features = self._extract_appearance_features(roi)
                if hist_features is not None:
                    self.appearance_history.append(hist_features)
        
        # Update height as a distinguishing feature
        self.last_height = height
        
        # Store position with high confidence as reliable reference
        if confidence > 0.7:
            self.last_reliable_position = (center_x, center_y)
        
        # Update position tracking
        position = (center_x, center_y)
        self.positions.append(position)
        self.long_term_positions.append(position)
        self.update_heatmap(position)
        
        self.keypoints_history.append(keypoints)
        self.confidence = confidence
        self.bbox = bbox
        self.keypoints = keypoints
        self.missing_frames = 0
        
    def _extract_appearance_features(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract appearance features from person ROI
        
        Args:
            roi: Region of interest (person crop)
            
        Returns:
            Feature vector or None if extraction failed
        """
        try:
            # Resize for consistency
            roi_resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
            
            # Calculate color histograms for each channel
            hist_features = []
            for channel in range(3):  # BGR channels
                hist = cv2.calcHist([roi_resized], [channel], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                hist_features.extend(hist)
                
            return np.array(hist_features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            return None
    
    def get_appearance_similarity(self, features: np.ndarray) -> float:
        """
        Compare new features with history to calculate appearance similarity
        
        Args:
            features: New appearance feature vector
            
        Returns:
            Similarity score (0-1)
        """
        if not self.appearance_history:
            return 0.0
            
        # Compare with average of recent appearances
        avg_features = np.mean(list(self.appearance_history), axis=0)
        
        # Compute cosine similarity
        dot_product = np.dot(avg_features, features)
        norm_a = np.linalg.norm(avg_features)
        norm_b = np.linalg.norm(features)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        similarity = dot_product / (norm_a * norm_b)
        return float(max(0.0, min(1.0, similarity)))  # Ensure range [0,1]
        
    def get_position(self) -> Optional[Point]:
        """Get current position if available"""
        if not self.positions:
            return None
        return self.positions[-1]
    
    def get_predicted_position(self, frames_ahead: int = 1) -> Optional[Point]:
        """
        Predict position based on motion model
        
        Args:
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted position (x, y) or None if prediction not possible
        """
        if len(self.positions) < 2:
            return self.get_position()
        
        # Get current position
        current_pos = self.get_position()
        if current_pos is None:
            return None
            
        # Use last reliable position if available and not too old
        if self.missing_frames > 10 and self.last_reliable_position:
            current_pos = self.last_reliable_position
        
        # Motion model using velocity and acceleration
        pred_x = current_pos[0] + (self.velocity[0] * frames_ahead) + (0.5 * self.acceleration[0] * frames_ahead**2)
        pred_y = current_pos[1] + (self.velocity[1] * frames_ahead) + (0.5 * self.acceleration[1] * frames_ahead**2)
        
        return (pred_x, pred_y)
    
    def mark_missing(self) -> None:
        """Mark player as missing in the current frame"""
        self.missing_frames += 1
        # Reduce ID confidence as frames missing increases
        self.id_confidence = max(0.1, self.id_confidence * 0.95)
    
    def calculate_matching_score(
        self, 
        detection: Detection, 
        frame: Optional[np.ndarray] = None,
        frame_dimensions: Optional[Tuple[int, int]] = None
    ) -> Dict[str, float]:
        """
        Calculate matching score for a detection using multiple cues
        
        Args:
            detection: Detection to calculate score for
            frame: Video frame for appearance comparison
            frame_dimensions: (width, height) for normalizing distances
            
        Returns:
            Dictionary with component scores and final score
        """
        scores = {
            'track_id': 0.0,
            'position': 0.0,
            'height': 0.0,
            'appearance': 0.0,
            'consistency': 0.0,
            'final': 0.0
        }
        
        # Track ID matching (strongest signal)
        if self.track_id is not None and detection.track_id == self.track_id:
            scores['track_id'] = 1.0
        
        # Position proximity score
        center = detection.center
        pred_pos = self.get_predicted_position()
        
        if pred_pos is not None and frame_dimensions is not None:
            width, height = frame_dimensions
            distance = np.sqrt((center[0] - pred_pos[0])**2 + (center[1] - pred_pos[1])**2)
            max_distance = np.sqrt(width**2 + height**2)
            scores['position'] = 1.0 - min(1.0, distance / (max_distance * 0.2))
        
        # Height similarity score
        if self.last_height is not None:
            height_diff = abs(detection.height - self.last_height) / max(self.last_height, 1)
            scores['height'] = 1.0 - min(1.0, height_diff)
        
        # Appearance similarity score
        if frame is not None:
            roi = detection.extract_roi(frame)
            if roi is not None and roi.size > 0:
                features = self._extract_appearance_features(roi)
                if features is not None:
                    scores['appearance'] = self.get_appearance_similarity(features)
        
        # Position consistency score
        scores['consistency'] = self.get_position_consistency_score(center)
        
        # Calculate weighted final score
        weights = {
            'track_id': 0.5,
            'position': 0.3,
            'appearance': 0.1, 
            'height': 0.05,
            'consistency': 0.05
        }
        
        final_score = sum(scores[k] * weights[k] for k in weights)
        
        # Adjust by ID confidence
        final_score = final_score * (0.5 + 0.5 * self.id_confidence)
        scores['final'] = final_score
        
        return scores


class KeypointProcessor:
    """
    Processes and manages keypoint data for pose estimation
    """
    
    # Standard keypoint names for COCO format used by YOLOv8 pose
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # Skeleton connections for visualization
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Face
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    @classmethod
    def process_keypoints(cls, keypoints: Optional[KeypointType]) -> Dict[str, Dict[str, float]]:
        """
        Process raw keypoints into a structured format
        
        Args:
            keypoints: Raw keypoints from model
            
        Returns:
            Dictionary of processed keypoints
        """
        processed = {}
        
        if keypoints is not None:
            for i, name in enumerate(cls.KEYPOINT_NAMES):
                if i < len(keypoints) and len(keypoints[i]) >= 3:
                    processed[name] = {
                        "x": float(keypoints[i][0]),
                        "y": float(keypoints[i][1]),
                        "confidence": float(keypoints[i][2])
                    }
                else:
                    processed[name] = {"x": 0, "y": 0, "confidence": 0}
        
        return processed
    
    @classmethod
    def draw_skeleton(cls, frame: np.ndarray, keypoints: KeypointType, color: Tuple[int, int, int], 
                    thickness: int = 2, circle_radius: int = 3) -> None:
        """
        Draw skeleton on frame
        
        Args:
            frame: Frame to draw on
            keypoints: Keypoints to visualize
            color: Color for skeleton (BGR)
            thickness: Line thickness
            circle_radius: Radius of keypoint circles
        """
        if keypoints is None:
            return
        
        # Draw skeleton connections
        for p1_idx, p2_idx in cls.SKELETON:
            if (p1_idx < len(keypoints) and p2_idx < len(keypoints) and
                keypoints[p1_idx][2] > 0.5 and keypoints[p2_idx][2] > 0.5):
                
                p1 = (int(keypoints[p1_idx][0]), int(keypoints[p1_idx][1]))
                p2 = (int(keypoints[p2_idx][0]), int(keypoints[p2_idx][1]))
                cv2.line(frame, p1, p2, color, thickness)
        
        # Draw keypoints
        for kp in keypoints:
            if kp[2] > 0.5:  # If confidence is high enough
                cv2.circle(frame, (int(kp[0]), int(kp[1])), circle_radius, color, -1)


class PlayerTrackingManager:
    """
    Manages tracking of multiple players, handling occlusions and identity switches
    """
    
    def __init__(
        self, 
        num_players: int = 2,
        max_missing_frames: int = 30,
        position_consistency_check_interval: int = 60, 
        swap_confidence_threshold: float = 0.85
    ):
        """
        Initialize tracking manager
        
        Args:
            num_players: Number of players to track
            max_missing_frames: Maximum frames to track a missing player
            position_consistency_check_interval: Frames between position consistency checks
            swap_confidence_threshold: Threshold for ID swap detection
        """
        self.players = {
            i+1: PlayerTracker(player_id=i+1) for i in range(num_players)
        }
        self.max_missing_frames = max_missing_frames
        self.position_consistency_check_interval = position_consistency_check_interval
        self.swap_confidence_threshold = swap_confidence_threshold
        self.initial_positions = {}
        self.frame_number = 0
        self.frame_dimensions = None
    
    def initialize_tracking(self, detections: List[Detection], frame: np.ndarray) -> None:
        """
        Initialize tracking with first frame detections
        
        Args:
            detections: List of detections
            frame: Video frame
        """
        if len(detections) < len(self.players):
            logger.warning(f"Not enough detections to initialize all players: {len(detections)} detections for {len(self.players)} players")
            return
        
        # Sort detections by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Initialize each player with a detection
        for i, (player_id, player) in enumerate(self.players.items()):
            if i < len(detections):
                player.update(detections[i], frame)
                player.track_id = detections[i].track_id
                self.initial_positions[player_id] = player.get_position()
        
        # Special case for initial player assignment in squash - make left player ID 1
        self._sort_players_horizontal()
        
        # Store frame dimensions for distance normalization
        self.frame_dimensions = (frame.shape[1], frame.shape[0])
    
    def _sort_players_horizontal(self) -> None:
        """Ensure player 1 is on the left side"""
        p1_pos = self.players[1].get_position()
        p2_pos = self.players[2].get_position()
        
        if p1_pos and p2_pos and p1_pos[0] > p2_pos[0]:
            logger.info("Initial setup: Swapping player IDs based on position")
            
            # Swap player data
            self._swap_player_data(1, 2)
            
            # Update initial positions
            self.initial_positions[1] = self.players[1].get_position()
            self.initial_positions[2] = self.players[2].get_position()
    
    def _swap_player_data(self, player1_id: int, player2_id: int) -> None:
        """
        Swap data between two players
        
        Args:
            player1_id: First player ID
            player2_id: Second player ID
        """
        p1 = self.players[player1_id]
        p2 = self.players[player2_id]
        
        # Attributes to swap
        attrs_to_swap = ['bbox', 'keypoints', 'confidence', 'track_id']
        
        for attr in attrs_to_swap:
            temp = getattr(p1, attr)
            setattr(p1, attr, getattr(p2, attr))
            setattr(p2, attr, temp)
    
    def check_for_id_swaps(self, frame: np.ndarray) -> bool:
        """
        Check if player IDs need to be swapped due to tracking errors
        
        Args:
            frame: Current video frame
            
        Returns:
            True if swap was performed
        """
        # Only check at specified intervals and after initial tracking is established
        if (self.frame_number <= 60 or 
            self.frame_number % self.position_consistency_check_interval != 0):
            return False
        
        # Only check if both players are visible
        p1 = self.players[1]
        p2 = self.players[2]
        
        if (p1.missing_frames > 0 or p2.missing_frames > 0 or 
            p1.bbox is None or p2.bbox is None):
            return False
        
        # Calculate position consistency scores
        player1_pos = p1.get_position()
        player2_pos = p2.get_position()
        
        # Check how consistent current positions are with historical distributions
        p1_at_p1_score = p1.get_position_consistency_score(player1_pos)
        p1_at_p2_score = p1.get_position_consistency_score(player2_pos)
        p2_at_p1_score = p2.get_position_consistency_score(player1_pos)
        p2_at_p2_score = p2.get_position_consistency_score(player2_pos)
        
        # Calculate swap likelihood score
        current_arrangement_score = p1_at_p1_score + p2_at_p2_score
        swapped_arrangement_score = p1_at_p2_score + p2_at_p1_score
        
        logger.info(f"Frame {self.frame_number}: Position consistency - Current: {current_arrangement_score:.2f}, Swapped: {swapped_arrangement_score:.2f}")
        
        # Detect potential swap
        if (swapped_arrangement_score > current_arrangement_score and 
            swapped_arrangement_score > self.swap_confidence_threshold):
            logger.warning(f"DETECTED ID SWAP at frame {self.frame_number}! Correcting...")
            
            # Swap tracking info
            self._swap_player_data(1, 2)
            return True
            
        return False
    
    def assign_detections_to_players(
        self, 
        detections: List[Detection], 
        frame: np.ndarray
    ) -> None:
        """
        Assign detections to players using multi-cue tracking
        
        Args:
            detections: List of detections
            frame: Current video frame
        """
        if not detections:
            # No detections, mark all players as missing
            for player in self.players.values():
                player.mark_missing()
            return
        
        # For the first frame, initialize tracking
        if self.frame_number == 0:
            self.initialize_tracking(detections, frame)
            return
        
        if len(detections) >= len(self.players):
            self._assign_multiple_detections(detections, frame)
        elif len(detections) == 1:
            self._assign_single_detection(detections[0], frame)
    
    def _assign_multiple_detections(self, detections: List[Detection], frame: np.ndarray) -> None:
        """
        Assign multiple detections to players
        
        Args:
            detections: List of detections
            frame: Current video frame
        """
        matching_scores = []
        
        # Calculate scores for all player-detection pairs
        for player_id, player in self.players.items():
            player_pos = player.get_position()
            if player_pos is None:
                continue
            
            for i, detection in enumerate(detections):
                # Calculate comprehensive matching score
                scores = player.calculate_matching_score(
                    detection, 
                    frame=frame,
                    frame_dimensions=self.frame_dimensions
                )
                
                # Store player-detection pair with score
                matching_scores.append((player_id, i, scores['final']))
        
        # Sort by score in descending order
        matching_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Assign detections to players
        assigned_detections: Set[int] = set()
        assigned_players: Set[int] = set()
        
        # First pass: assign only high confidence matches
        high_confidence_threshold = 0.6
        
        for player_id, detection_idx, score in matching_scores:
            # Skip if already assigned
            if detection_idx in assigned_detections or player_id in assigned_players:
                continue
            
            if score > high_confidence_threshold:
                # Update player with this detection
                detection = detections[detection_idx]
                player = self.players[player_id]
                player.update(detection, frame)
                player.track_id = detection.track_id
                player.id_confidence = min(1.0, player.id_confidence + 0.1)
                
                assigned_detections.add(detection_idx)
                assigned_players.add(player_id)
        
        # Second pass: assign remaining detections
        for player_id, detection_idx, score in matching_scores:
            if detection_idx in assigned_detections or player_id in assigned_players:
                continue
            
            # Update player with this detection
            detection = detections[detection_idx]
            player = self.players[player_id]
            player.update(detection, frame)
            player.track_id = detection.track_id
            
            # Lower confidence for less certain assignments
            player.id_confidence = max(0.1, min(0.8, score))
            
            assigned_detections.add(detection_idx)
            assigned_players.add(player_id)
            
            # If all players are assigned, break
            if len(assigned_players) >= len(self.players):
                break
        
        # Mark unassigned players as missing
        for player_id, player in self.players.items():
            if player_id not in assigned_players:
                if player.missing_frames == 0:  # Just started missing
                    logger.info(f"Player {player_id} missing at frame {self.frame_number}")
                player.mark_missing()
    
    def _assign_single_detection(self, detection: Detection, frame: np.ndarray) -> None:
        """
        Assign a single detection to the most likely player
        
        Args:
            detection: Detection to assign
            frame: Current video frame
        """
        # Calculate scores for each player
        player_scores = []
        
        for player_id, player in self.players.items():
            # Track ID match (highest priority)
            if player.track_id is not None and detection.track_id == player.track_id:
                player_scores.append((player_id, 0.9))
                continue
            
            # Calculate comprehensive matching score
            scores = player.calculate_matching_score(
                detection, 
                frame=frame,
                frame_dimensions=self.frame_dimensions
            )
            
            player_scores.append((player_id, scores['final']))
        
        # Assign to player with highest score
        if player_scores:
            player_scores.sort(key=lambda x: x[1], reverse=True)
            best_player_id, best_score = player_scores[0]
            
            # Only assign if score is reasonable
            assignment_threshold = 0.3
            if best_score > assignment_threshold:
                # Update best matching player
                self.players[best_player_id].update(detection, frame)
                self.players[best_player_id].track_id = detection.track_id
                
                # Mark other players as missing
                for player_id in self.players:
                    if player_id != best_player_id:
                        self.players[player_id].mark_missing()
            else:
                # Score too low, mark all as missing
                logger.warning(f"Frame {self.frame_number}: Detection score too low ({best_score:.2f}), marking all players as missing")
                for player in self.players.values():
                    player.mark_missing()
        else:
            # No scores calculated (shouldn't happen), fallback to player 1
            logger.warning(f"Frame {self.frame_number}: No scores calculated, assigning to player 1")
            self.players[1].update(detection, frame)
            self.players[1].track_id = detection.track_id
            
            # Mark other players as missing
            for player_id in range(2, len(self.players) + 1):
                self.players[player_id].mark_missing()
    
    def visualize_tracking(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracking visualization on frame
        
        Args:
            frame: Frame to visualize tracking on
            
        Returns:
            Frame with visualizations
        """
        display_frame = frame.copy()
        
        # Draw predicted positions for missing players
        for player_id, player in self.players.items():
            if 0 < player.missing_frames <= self.max_missing_frames:
                predicted_pos = player.get_predicted_position(player.missing_frames)
                
                if predicted_pos is not None:
                    # Draw predicted position with a different style
                    cv2.circle(display_frame, 
                              (int(predicted_pos[0]), int(predicted_pos[1])), 
                              10, player.color, 2)
                    cv2.putText(display_frame, f"P{player_id} (predicted)", 
                                (int(predicted_pos[0]) + 10, int(predicted_pos[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, player.color, 2)
        
        # Draw tracking info for visible players
        for player_id, player in self.players.items():
            # Skip if player is missing for too long
            if player.missing_frames > self.max_missing_frames:
                continue
                
            if player.bbox is not None and player.missing_frames == 0:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, player.bbox)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), player.color, 2)
                
                # Draw label
                cv2.putText(display_frame, f"Player {player_id} ({player.confidence:.2f})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, player.color, 2)
                
                # Draw keypoints if available
                if player.keypoints is not None:
                    KeypointProcessor.draw_skeleton(display_frame, player.keypoints, player.color)
        
        # Add ID confidence information to the display
        for player_id, player in self.players.items():
            if player.missing_frames <= self.max_missing_frames:
                confidence_text = f"P{player_id} ID conf: {player.id_confidence:.2f}"
                y_pos = 60 + 30 * player_id
                cv2.putText(display_frame, confidence_text, (10, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, player.color, 2)
        
        # Add frame counter
        cv2.putText(display_frame, f"Frame: {self.frame_number}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame
    
    def update(self, detections: List[Detection], frame: np.ndarray) -> np.ndarray:
        """
        Process a new frame
        
        Args:
            detections: List of detections in the frame
            frame: Current video frame
            
        Returns:
            Visualization frame
        """
        # Assign detections to players
        self.assign_detections_to_players(detections, frame)
        
        # Check for ID swaps and correct if needed
        swap_detected = self.check_for_id_swaps(frame)
        
        # Create visualization
        display_frame = self.visualize_tracking(frame)
        
        # Add swap notification if detected
        if swap_detected:
            h, w = frame.shape[:2]
            cv2.putText(display_frame, "ID SWAP CORRECTED", (w//2 - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Increment frame counter
        self.frame_number += 1
        
        return display_frame


class SquashPlayerTracker:
    """
    Main class for tracking squash players in video
    """
    
    def __init__(
        self,
        model_path: str = "models/yolo11m-pose.pt",
        confidence_threshold: float = 0.35,
        max_missing_frames: int = 30,
        use_gpu: bool = True
    ):
        """
        Initialize the tracker
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
            max_missing_frames: Maximum frames to track a missing player
            use_gpu: Whether to use GPU for model inference
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.max_missing_frames = max_missing_frames
        
        # Initialize YOLO model
        self.model = self._initialize_model(use_gpu)
        
        # Initialize tracking manager
        self.tracking_manager = PlayerTrackingManager(
            num_players=2,
            max_missing_frames=max_missing_frames
        )
        
        # Data collection
        self.csv_writer = None
        self.keypoint_names = KeypointProcessor.KEYPOINT_NAMES
    
    def _initialize_model(self, use_gpu: bool) -> YOLO:
        """
        Initialize YOLO model
        
        Args:
            use_gpu: Whether to use GPU
            
        Returns:
            Initialized YOLO model
        """
        logger.info(f"Loading YOLO model from {self.model_path}")
        model = YOLO(self.model_path)
        
        # Set device
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        model.to(device)
        
        return model
    
    def _process_detections(self, results: Any) -> List[Detection]:
        """
        Process YOLO results into Detection objects
        
        Args:
            results: YOLO model results
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        if results and len(results[0]) > 0:
            # Extract boxes, keypoints and tracking IDs
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            
            # Get tracking IDs if available
            if hasattr(boxes, 'id') and boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)
            
            # Process each detection
            for i in range(len(boxes)):
                # Get bounding box
                box = boxes[i].xyxy.cpu().numpy()[0]
                confidence = float(boxes[i].conf.cpu().numpy()[0])
                track_id = track_ids[i] if i < len(track_ids) else None
                
                # Get keypoints if available
                kpts = None
                if keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i].data.cpu().numpy()[0]
                
                # Create Detection object
                detection = Detection(
                    bbox=tuple(box),
                    confidence=confidence,
                    keypoints=kpts,
                    track_id=track_id
                )
                
                detections.append(detection)
        
        return detections
    
    def _initialize_csv(self, csv_path: str) -> None:
        """
        Initialize CSV writer for tracking data
        
        Args:
            csv_path: Path to CSV file
        """
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        # Create header
        header = ['frame', 'time_sec', 'player_id']
        
        # Add keypoint columns
        for kp in self.keypoint_names:
            header.extend([f"{kp}_x", f"{kp}_y", f"{kp}_conf"])
        
        # Add bounding box and confidence columns
        header.extend(['center_x', 'center_y', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'detection_confidence'])
        
        # Write header
        csv_writer.writerow(header)
        
        self.csv_writer = csv_writer
        self.csv_file = csv_file
    
    def _write_player_data(self, frame_number: int, time_sec: float) -> None:
        """
        Write player data to CSV
        
        Args:
            frame_number: Current frame number
            time_sec: Time in seconds
        """
        if self.csv_writer is None:
            return
            
        for player_id, player in self.tracking_manager.players.items():
            # Skip if player has no keypoints
            if player.keypoints is None or player.bbox is None or player.missing_frames > 0:
                continue
                
            # Process keypoints
            processed_keypoints = KeypointProcessor.process_keypoints(player.keypoints)
            
            # Prepare CSV row
            row = [frame_number, time_sec, player_id]
            
            # Add all keypoint data
            for kp_name in self.keypoint_names:
                kp_data = processed_keypoints.get(kp_name, {"x": 0, "y": 0, "confidence": 0})
                row.extend([kp_data["x"], kp_data["y"], kp_data["confidence"]])
            
            # Calculate center position
            x1, y1, x2, y2 = player.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Add center position and bbox
            row.extend([center_x, center_y, x1, y1, x2, y2, player.confidence])
            
            # Write to CSV
            self.csv_writer.writerow(row)
    
    def process_video(self, video_path: str, output_dir: str) -> Tuple[str, str]:
        """
        Process a video and track players
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save output
            
        Returns:
            Tuple of (output_video_path, csv_path)
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{video_name}_tracked.mp4")
        csv_path = os.path.join(output_dir, "player_tracking_data.csv")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Initialize CSV writer
        self._initialize_csv(csv_path)
        
        # Process video frame by frame
        frame_number = 0
        logger.info(f"Processing video: {video_path} ({total_frames} frames)")
        
        # Use tqdm for progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Time in seconds
                time_sec = frame_number / fps
                
                # Run YOLO detection with pose estimation
                results = self.model.track(
                    frame, 
                    conf=self.confidence_threshold,
                    persist=True,  # Enable tracking
                    verbose=False,
                    classes=0      # Class 0 is person
                )
                
                # Process detections
                detections = self._process_detections(results)
                
                # Update tracking manager
                display_frame = self.tracking_manager.update(detections, frame)
                
                # Write player data to CSV
                self._write_player_data(frame_number, time_sec)
                
                # Write frame to output video
                out.write(display_frame)
                
                # Display frame
                cv2.imshow("Player Tracking", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                frame_number += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        self.csv_file.close()
        cv2.destroyAllWindows()
        
        logger.info(f"Processing complete!")
        logger.info(f"Output video saved to: {output_video_path}")
        logger.info(f"Player tracking data saved to: {csv_path}")
        
        return output_video_path, csv_path


class Visualizer:
    """
    Create visualizations from tracking data
    """
    
    @staticmethod
    def create_player_heatmaps(
        csv_path: str, 
        output_dir: str, 
        width: int, 
        height: int
    ) -> str:
        """
        Create heatmap visualization of player positions
        
        Args:
            csv_path: Path to CSV with tracking data
            output_dir: Directory to save output
            width: Video width
            height: Video height
            
        Returns:
            Path to saved heatmap image
        """
        # Read data from CSV
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Get header
            for row in reader:
                data.append(row)
        
        if not data:
            logger.warning("No data to visualize")
            return ""
        
        # Convert to numpy array for easier processing
        data = np.array(data)
        
        # Split by player
        player1_data = data[data[:, 2] == '1']
        player2_data = data[data[:, 2] == '2']
        
        # Create figure with better styling
        plt.figure(figsize=(12, 10))
        plt.style.use('ggplot')
        plt.title("Player Position Heatmap", fontsize=16)
        
        # Add court outline (simplified)
        plt.axhline(height/2, color='white', linestyle='--', alpha=0.4)
        plt.axvline(width/2, color='white', linestyle='--', alpha=0.4)
        
        # Player 1 (red)
        x_idx = header.index('center_x')
        y_idx = header.index('center_y')
        if len(player1_data) > 0:
            plt.hexbin(
                player1_data[:, x_idx].astype(float), 
                player1_data[:, y_idx].astype(float),
                gridsize=75, cmap='Reds', alpha=0.8,
                extent=(0, width, height, 0),
                label='Player 1'
            )
        
        # Player 2 (blue)
        if len(player2_data) > 0:
            plt.hexbin(
                player2_data[:, x_idx].astype(float), 
                player2_data[:, y_idx].astype(float),
                gridsize=75, cmap='Blues', alpha=0.8,
                extent=(0, width, height, 0),
                label='Player 2'
            )
        
        plt.xlabel("X Position", fontsize=12)
        plt.ylabel("Y Position", fontsize=12)
        plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
        
        # Add colorbar and legend
        plt.colorbar(label='Position Density')
        
        # Save with high quality
        heatmap_path = os.path.join(output_dir, "player_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to: {heatmap_path}")
        
        plt.close()
        return heatmap_path
    
    @staticmethod
    def create_movement_plot(
        csv_path: str, 
        output_dir: str
    ) -> str:
        """
        Create plot of player movement over time
        
        Args:
            csv_path: Path to CSV with tracking data
            output_dir: Directory to save output
            
        Returns:
            Path to saved movement plot
        """
        # Read data from CSV
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Get header
            for row in reader:
                data.append(row)
        
        if not data:
            logger.warning("No data to visualize")
            return ""
        
        # Convert to numpy array for easier processing
        data = np.array(data)
        
        # Split by player
        player1_data = data[data[:, 2] == '1']
        player2_data = data[data[:, 2] == '2']
        
        # Create figure with better styling
        plt.figure(figsize=(14, 8))
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.title("Player Movement Over Time", fontsize=16)
        
        time_idx = header.index('time_sec')
        x_idx = header.index('center_x')
        y_idx = header.index('center_y')
        
        # Apply smoothing
        def smooth(y, window_size=5):
            box = np.ones(window_size) / window_size
            return np.convolve(y, box, mode='same')
        
        # Player 1 movement
        if len(player1_data) > 0:
            times = player1_data[:, time_idx].astype(float)
            
            # X movement (smoothed)
            x_pos = player1_data[:, x_idx].astype(float)
            if len(x_pos) > 5:
                x_pos_smooth = smooth(x_pos)
                plt.plot(times, x_pos_smooth, 'r-', linewidth=2, label='Player 1 X', alpha=0.8)
            else:
                plt.plot(times, x_pos, 'r-', linewidth=2, label='Player 1 X', alpha=0.8)
            
            # Y movement (smoothed)
            y_pos = player1_data[:, y_idx].astype(float)
            if len(y_pos) > 5:
                y_pos_smooth = smooth(y_pos)
                plt.plot(times, y_pos_smooth, 'r--', linewidth=2, label='Player 1 Y', alpha=0.8)
            else:
                plt.plot(times, y_pos, 'r--', linewidth=2, label='Player 1 Y', alpha=0.8)
        
        # Player 2 movement
        if len(player2_data) > 0:
            times = player2_data[:, time_idx].astype(float)
            
            # X movement (smoothed)
            x_pos = player2_data[:, x_idx].astype(float)
            if len(x_pos) > 5:
                x_pos_smooth = smooth(x_pos)
                plt.plot(times, x_pos_smooth, 'b-', linewidth=2, label='Player 2 X', alpha=0.8)
            else:
                plt.plot(times, x_pos, 'b-', linewidth=2, label='Player 2 X', alpha=0.8)
            
            # Y movement (smoothed)
            y_pos = player2_data[:, y_idx].astype(float)
            if len(y_pos) > 5:
                y_pos_smooth = smooth(y_pos)
                plt.plot(times, y_pos_smooth, 'b--', linewidth=2, label='Player 2 Y', alpha=0.8)
            else:
                plt.plot(times, y_pos, 'b--', linewidth=2, label='Player 2 Y', alpha=0.8)
        
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Position (pixels)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save with high quality
        movement_path = os.path.join(output_dir, "player_movement.png")
        plt.savefig(movement_path, dpi=300, bbox_inches='tight')
        logger.info(f"Movement plot saved to: {movement_path}")
        
        plt.close()
        return movement_path
    
    @staticmethod
    def create_visualizations(
        csv_path: str, 
        output_dir: str, 
        width: int, 
        height: int
    ) -> Dict[str, str]:
        """
        Create all visualizations
        
        Args:
            csv_path: Path to CSV with tracking data
            output_dir: Directory to save output
            width: Video width
            height: Video height
            
        Returns:
            Dictionary of visualization paths
        """
        visualizations = {}
        
        # Create heatmap
        visualizations['heatmap'] = Visualizer.create_player_heatmaps(
            csv_path, output_dir, width, height
        )
        
        # Create movement plot
        visualizations['movement'] = Visualizer.create_movement_plot(
            csv_path, output_dir
        )
        
        return visualizations


def main():
    """Main entry point"""
    # Set up command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Track squash players in video')
    parser.add_argument('--video', type=str, default="farag_elshorbagy_1m_chopped.mp4",
                       help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: timestamped directory)')
    parser.add_argument('--model', type=str, default="models/yolo11m-pose.pt",
                       help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.35,
                       help='Detection confidence threshold')
    parser.add_argument('--missing-frames', type=int, default=30,
                       help='Maximum frames to track missing players')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU inference')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("tracking_output", f"player_tracking_{timestamp}")
    else:
        output_dir = args.output
    
    # Initialize tracker
    tracker = SquashPlayerTracker(
        model_path=args.model,
        confidence_threshold=args.conf,
        max_missing_frames=args.missing_frames,
        use_gpu=not args.cpu
    )
    
    # Process video
    output_video, csv_path = tracker.process_video(
        video_path=args.video,
        output_dir=output_dir
    )
    
    # Get video properties for visualization
    cap = cv2.VideoCapture(args.video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Create visualizations
    Visualizer.create_visualizations(csv_path, output_dir, width, height)
    
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

