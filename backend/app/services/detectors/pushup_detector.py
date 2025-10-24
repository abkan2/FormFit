# Create: app/services/detectors/pushup_detector.py

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import time

class PushupDetector:
    """Rule-based pushup detection optimized for reliability"""
    
    def __init__(self):
        self.rep_count = 0
        self.current_phase = "neutral"
        self.last_phase = "neutral"
        self.phase_history = deque(maxlen=10)
        self.last_rep_time = 0
        self.min_rep_interval = 1.0  # Minimum seconds between reps
        self.position_history = deque(maxlen=5)
        
    def detect_pushup_phase(self, landmarks: List[Tuple[float, float]]) -> str:
        """Detect pushup phase using reliable geometric rules"""
        if len(landmarks) != 33:
            return "neutral"
        
        # Key landmarks
        left_wrist = landmarks[15]
        right_wrist = landmarks[16] 
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        nose = landmarks[0]
        
        # Calculate positions
        avg_wrist_y = (left_wrist[1] + right_wrist[1]) / 2
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        avg_elbow_y = (left_elbow[1] + right_elbow[1]) / 2
        nose_y = nose[1]
        
        # Multiple detection methods for robustness
        
        # Method 1: Wrist-shoulder relationship
        wrist_shoulder_diff = avg_wrist_y - avg_shoulder_y
        
        # Method 2: Elbow angle approximation
        elbow_shoulder_diff = avg_elbow_y - avg_shoulder_y
        
        # Method 3: Overall body position (nose relative to shoulders)
        nose_shoulder_diff = nose_y - avg_shoulder_y
        
        # Combine methods for robust detection
        up_indicators = 0
        down_indicators = 0
        
        # Indicator 1: Wrists above shoulders (strong up signal)
        if wrist_shoulder_diff < -0.15:
            up_indicators += 2
        elif wrist_shoulder_diff > 0.1:
            down_indicators += 1
            
        # Indicator 2: Elbows position
        if elbow_shoulder_diff < -0.05:
            up_indicators += 1
        elif elbow_shoulder_diff > 0.05:
            down_indicators += 1
            
        # Indicator 3: Head position (pushup form)
        if nose_shoulder_diff < -0.1:
            up_indicators += 1
        elif nose_shoulder_diff > 0.05:
            down_indicators += 1
        
        # Decision logic
        if up_indicators >= 2:
            phase = "up"
        elif down_indicators >= 2:
            phase = "down"
        else:
            phase = "transition"
            
        # Smooth the detection
        self.position_history.append(phase)
        
        # Use majority vote from recent history for stability
        if len(self.position_history) >= 3:
            recent_phases = list(self.position_history)[-3:]
            phase_counts = {p: recent_phases.count(p) for p in set(recent_phases)}
            phase = max(phase_counts, key=phase_counts.get)
        
        return phase
    
    def update(self, landmarks: List[Tuple[float, float]]) -> dict:
        """Update detector state and check for rep completion"""
        self.last_phase = self.current_phase
        self.current_phase = self.detect_pushup_phase(landmarks)
        self.phase_history.append(self.current_phase)
        
        # Check for rep completion
        rep_completed = self.check_rep_completion()
        
        return {
            'phase': self.current_phase,
            'rep_count': self.rep_count,
            'rep_completed': rep_completed,
            'phase_history': list(self.phase_history)
        }
    
    def check_rep_completion(self) -> bool:
        """Detect completed rep using phase transitions"""
        if len(self.phase_history) < 4:
            return False
            
        current_time = time.time()
        if current_time - self.last_rep_time < self.min_rep_interval:
            return False
        
        # Look for down->up transition pattern
        recent = list(self.phase_history)[-6:]  # Last 6 phases
        
        # Pattern matching for rep completion
        has_down = "down" in recent[:-1]  # Had down position
        is_up_now = recent[-1] == "up"    # Currently up
        
        # Additional validation: ensure we had a proper transition
        if has_down and is_up_now:
            # Count transitions to ensure it's a complete movement
            transitions = 0
            for i in range(1, len(recent)):
                if recent[i] != recent[i-1]:
                    transitions += 1
            
            # Valid rep needs at least 2 transitions (down->transition->up minimum)
            if transitions >= 2:
                self.count_rep()
                return True
        
        return False
    
    def count_rep(self):
        """Register a completed rep"""
        self.rep_count += 1
        self.last_rep_time = time.time()
        print(f"🎉 Rep {self.rep_count} completed!")
        
        # Clear recent history to avoid double counting
        self.phase_history.clear()
        
    def reset(self):
        """Reset counter"""
        self.rep_count = 0
        self.phase_history.clear()
        self.position_history.clear()
        self.last_rep_time = 0