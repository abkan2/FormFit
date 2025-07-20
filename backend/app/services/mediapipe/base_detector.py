# backend/app/services/mediapipe/base_detector.py

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, Dict
import logging

class BasePoseDetector:
    def __init__(self, 
                 static_image_mode=False, 
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Enhanced pose detector with more configuration options
        
        Args:
            model_complexity: 0, 1, or 2. Higher = more accurate but slower
            smooth_landmarks: Whether to smooth landmarks across frames
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Key landmarks for fitness exercises
        self.key_landmarks = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'nose': 0, 'left_eye': 2, 'right_eye': 5
        }
        
        self.logger = logging.getLogger(__name__)
    
    def process(self, image: np.ndarray) -> Optional[object]:
        """
        Process image and return pose landmarks
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Pose landmarks object or None if no pose detected
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            return results.pose_landmarks
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return None
    
    def extract_coordinates(self, landmarks) -> List[Tuple[float, float]]:
        """Extract normalized (x, y) coordinates from landmarks"""
        if not landmarks:
            return []
        return [(lm.x, lm.y) for lm in landmarks.landmark]
    
    def extract_key_points(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """Extract key landmarks relevant for fitness exercises"""
        if not landmarks:
            return {}
        
        key_points = {}
        for name, idx in self.key_landmarks.items():
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                key_points[name] = (lm.x, lm.y, lm.z, lm.visibility)
        return key_points
    
    def calculate_angles(self, landmarks) -> Dict[str, float]:
        """Calculate key joint angles for exercise analysis"""
        if not landmarks:
            return {}
        
        def get_angle(p1, p2, p3):
            """Calculate angle between three points"""
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)
            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            return 360 - angle if angle > 180.0 else angle
        
        try:
            points = self.extract_key_points(landmarks)
            angles = {}
            
            # Left arm angle (shoulder-elbow-wrist)
            if all(k in points for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
                angles['left_arm'] = get_angle(
                    points['left_shoulder'][:2],
                    points['left_elbow'][:2],
                    points['left_wrist'][:2]
                )
            
            # Right arm angle
            if all(k in points for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                angles['right_arm'] = get_angle(
                    points['right_shoulder'][:2],
                    points['right_elbow'][:2],
                    points['right_wrist'][:2]
                )
            
            # Left leg angle (hip-knee-ankle)
            if all(k in points for k in ['left_hip', 'left_knee', 'left_ankle']):
                angles['left_leg'] = get_angle(
                    points['left_hip'][:2],
                    points['left_knee'][:2],
                    points['left_ankle'][:2]
                )
                    
            # Right leg angle
            if all(k in points for k in ['right_hip', 'right_knee', 'right_ankle']):
                angles['right_leg'] = get_angle(
                    points['right_hip'][:2],
                    points['right_knee'][:2],
                    points['right_ankle'][:2]
                )
            
            # Torso angle (vertical reference)
            if all(k in points for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                shoulder_center = (
                    (points['left_shoulder'][0] + points['right_shoulder'][0]) / 2,
                    (points['left_shoulder'][1] + points['right_shoulder'][1]) / 2
                )
                hip_center = (
                    (points['left_hip'][0] + points['right_hip'][0]) / 2,
                    (points['left_hip'][1] + points['right_hip'][1]) / 2
                )
                # Angle from vertical
                angles['torso'] = np.abs(np.arctan2(
                    shoulder_center[0] - hip_center[0],
                    hip_center[1] - shoulder_center[1]
                ) * 180.0 / np.pi)
                
            return angles
            
        except Exception as e:
            self.logger.error(f"Error calculating angles: {e}")
            return {}
    
    def draw_landmarks(self, image: np.ndarray, landmarks) -> np.ndarray:
        """Draw pose landmarks on image"""
        if landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return image
    
    def is_pose_complete(self, landmarks, required_visibility=0.5) -> bool:
        """Check if pose has sufficient visible landmarks"""
        if not landmarks:
            return False
        
        visible_count = sum(1 for lm in landmarks.landmark 
                          if lm.visibility > required_visibility)
        return visible_count >= len(self.key_landmarks) * 0.7  # At least 70% visible
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'pose'):
            self.pose.close()