# backend/app/services/mediapipe/base_detector.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple
import logging

class BasePoseDetector:
    """Optimized pose detector for ML-based movement detection"""
    
    def __init__(self, 
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        """
        Simplified initialization focused on performance for ML inference
        
        Args:
            min_detection_confidence: Higher threshold for better accuracy
            min_tracking_confidence: Confidence for landmark tracking
        """
        self.mp_pose = mp.solutions.pose
        
        # Optimized settings for real-time ML inference
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,              # Good balance of speed/accuracy
            smooth_landmarks=True,           # Important for temporal sequences
            enable_segmentation=False,       # Not needed, saves processing
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Only the landmarks we actually use in ML training
        self.ml_landmarks = [
            11, 12,  # shoulders
            13, 14,  # elbows
            15, 16,  # wrists
            23, 24,  # hips
            25, 26,  # knees
            27, 28   # ankles
        ]
        
        self.logger = logging.getLogger(__name__)
    
    def process(self, image: np.ndarray) -> Optional[object]:
        """
        Process image and return pose landmarks (optimized for speed)
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Pose landmarks object or None if no pose detected
        """
        try:
            # Convert BGR to RGB (required by MediaPipe)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_image)
            
            # Return landmarks if pose detected with good confidence
            if results.pose_landmarks:
                # Quick visibility check for key landmarks
                key_visible = sum(1 for i in self.ml_landmarks 
                                if results.pose_landmarks.landmark[i].visibility > 0.5)
                
                if key_visible >= len(self.ml_landmarks) * 0.6:  # At least 60% visible
                    return results.pose_landmarks
            
            return None
            
        except Exception as e:
            self.logger.error(f"MediaPipe processing error: {e}")
            return None
    
    def extract_ml_coordinates(self, landmarks) -> List[Tuple[float, float]]:
        """
        Extract normalized (x, y) coordinates optimized for ML model
        Returns exactly what your ML model expects: 33 landmarks as (x, y) pairs
        """
        if not landmarks:
            return []
        
        # Return all 33 landmarks as (x, y) tuples - matches your training data
        return [(lm.x, lm.y) for lm in landmarks.landmark]
    
    def get_landmark_quality(self, landmarks) -> float:
        """
        Get overall quality score of detected landmarks
        Used to filter out poor detections before ML processing
        """
        if not landmarks:
            return 0.0
        
        # Calculate average visibility of key landmarks
        key_visibilities = [landmarks.landmark[i].visibility for i in self.ml_landmarks]
        return sum(key_visibilities) / len(key_visibilities)
    
    def draw_minimal_landmarks(self, image: np.ndarray, landmarks) -> np.ndarray:
        """
        Draw minimal landmarks for debugging (only key points for ML)
        Much faster than full landmark drawing
        """
        if not landmarks:
            return image
        
        h, w = image.shape[:2]
        
        # Draw only key landmarks as circles
        for idx in self.ml_landmarks:
            landmark = landmarks.landmark[idx]
            if landmark.visibility > 0.5:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
        
        # Draw key connections for pushups (arms and torso)
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (12, 24),  # Torso sides
            (23, 24)             # Hip line
        ]
        
        for start_idx, end_idx in connections:
            start_lm = landmarks.landmark[start_idx]
            end_lm = landmarks.landmark[end_idx]
            
            if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(image, start_pos, end_pos, (255, 0, 0), 2)
        
        return image
    
    def close(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()