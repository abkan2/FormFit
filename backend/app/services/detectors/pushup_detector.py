import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
import time

class PushupDetector:
    """Calibration-aware pushup detection for personalized accuracy"""
    
    def __init__(self, calibrator=None):
        self.calibrator = calibrator
        self.is_calibrated = calibrator is not None and calibrator.is_calibrated
        
        # Rep counting
        self.rep_count = 0
        self.current_phase = "neutral"
        self.last_phase = "neutral"
        self.phase_history = deque(maxlen=10)
        self.last_rep_time = 0
        self.min_rep_interval = 0.5
        
        # Phase tracking
        self.phase_start_time = time.time()
        self.time_in_phase = 0

        #Smoothing for velocity tracking
        self.raw_phase_history = deque(maxlen=5)
        self.phase_confidence = 0

        self.elbow_angle_history = deque(maxlen=5)
        self.last_elbow_angle = None
        
        # Form tracking
        self.form_issues = []
        self.good_form_count = 0
        self.total_reps = 0
        
        # ✅ CHANGE 3: Initialize thresholds based on calibration
        if self.is_calibrated:
            self._initialize_calibrated_thresholds()
        else:
            self._initialize_default_thresholds()
    
    def _initialize_calibrated_thresholds(self):
        """✅ THIS IS WHERE CALIBRATION DATA IS USED!"""
        # ✅ Get user's ACTUAL body measurements from calibration
        shoulder_width = self.calibrator.get_measurement('shoulder_width')
        torso_length = self.calibrator.get_measurement('torso_length')
        arm_span = self.calibrator.get_measurement('arm_span')
        
        print(f"🎯 Initializing CALIBRATED push-up detector")
        print(f"   📏 YOUR shoulder width: {shoulder_width:.3f}")
        print(f"   📏 YOUR torso length: {torso_length:.3f}")
        print(f"   📏 YOUR arm span: {arm_span:.3f}")
        
        # ✅ PERSONALIZED: Thresholds based on YOUR body
        # Down position threshold = 15% of YOUR torso length
        self.down_threshold_ratio = 0.15
        self.down_threshold = torso_length * self.down_threshold_ratio
        
        # ✅ PERSONALIZED: Body alignment based on YOUR measurements
        self.body_alignment_tolerance = torso_length * 0.2
        
        # ✅ PERSONALIZED: Hip sag detection for YOUR body
        self.max_hip_sag_ratio = 0.15
        
        # Elbow angles (same for everyone)
        self.down_elbow_angle = 90
        self.up_elbow_angle = 160
        
    
    def _initialize_default_thresholds(self):
        """❌ Generic thresholds (used when NO calibration)"""
        print("⚠️ No calibration - using GENERIC thresholds")
        print("   💡 Results will be less accurate for your body type!")
        
        self.down_threshold = 0.05  # Generic - doesn't fit anyone perfectly
        self.body_alignment_tolerance = 0.08
        self.max_hip_sag_ratio = 0.15
        self.down_elbow_angle = 90
        self.up_elbow_angle = 160
    
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """Calculate angle at point2 formed by three points"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def check_body_alignment(self, landmarks: List[Tuple[float, float]]) -> Tuple[bool, List[str]]:
        """✅ Check alignment using CALIBRATED thresholds"""
        issues = []
        
        try:
            # Key points
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            
            # Average positions
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            hip_y = (left_hip[1] + right_hip[1]) / 2
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            
            # ✅ CALIBRATED: Check hip sag using YOUR body measurements
            if self.is_calibrated:
                torso_length = self.calibrator.get_measurement('torso_length')
                
                # Hip sag relative to YOUR torso length
                hip_sag = hip_y - ((shoulder_y + ankle_y) / 2)
                max_sag = torso_length * self.max_hip_sag_ratio
                
                if hip_sag > max_sag:
                    issues.append("hips_sagging")
                elif hip_sag < -max_sag:
                    issues.append("hips_too_high")
            else:
                # ❌ Generic check (not personalized)
                hip_sag = hip_y - ((shoulder_y + ankle_y) / 2)
                if hip_sag > 0.05:
                    issues.append("hips_sagging")
                elif hip_sag < -0.05:
                    issues.append("hips_too_high")
            
            # Check body straightness using calibrated tolerance
            if abs(hip_y - ((shoulder_y + ankle_y) / 2)) > self.body_alignment_tolerance:
                if "hips_sagging" not in issues and "hips_too_high" not in issues:
                    issues.append("body_not_straight")
            
            is_aligned = len(issues) == 0
            return is_aligned, issues
            
        except (IndexError, TypeError):
            return False, ["landmark_error"]
    
    def detect_pushup_phase(self, landmarks: List[Tuple[float, float]]) -> Tuple[str, Dict]:
        """✅ IMPROVED: Velocity-aware phase detection"""
        try:
            # Key landmarks
            nose = landmarks[0]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Calculate elbow angles
            left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
            
            # ✅ NEW: Track velocity (angle change rate)
            elbow_velocity = 0
            if self.last_elbow_angle is not None:
                elbow_velocity = avg_elbow_angle - self.last_elbow_angle
            
            self.elbow_angle_history.append(avg_elbow_angle)
            self.last_elbow_angle = avg_elbow_angle
            
            # ✅ NEW: Smooth angle using moving average
            if len(self.elbow_angle_history) >= 3:
                smoothed_angle = np.mean(list(self.elbow_angle_history)[-3:])
            else:
                smoothed_angle = avg_elbow_angle
            
            # Calculate normalized distance
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            nose_shoulder_dist = abs(nose[1] - shoulder_y)
            
            if self.is_calibrated:
                torso_length = self.calibrator.get_measurement('torso_length')
                normalized_dist = nose_shoulder_dist / torso_length
            else:
                normalized_dist = nose_shoulder_dist
            
            # Check body alignment
            is_aligned, alignment_issues = self.check_body_alignment(landmarks)
            
            phase_data = {
                'elbow_angle': avg_elbow_angle,
                'smoothed_angle': smoothed_angle,
                'elbow_velocity': elbow_velocity,
                'nose_shoulder_dist': nose_shoulder_dist,
                'normalized_dist': normalized_dist,
                'is_aligned': is_aligned,
                'alignment_issues': alignment_issues
            }
            
            # ✅ IMPROVED: More lenient thresholds with velocity awareness
            # Down position: bent elbows
            if smoothed_angle < 100:  # ✅ Using smoothed angle
                raw_phase = "down" if is_aligned else "down_bad_form"
            
            # Up position: straight elbows  
            elif smoothed_angle > 145:  # ✅ Lowered from 150 for faster reps
                raw_phase = "up" if is_aligned else "up_bad_form"
            
            # ✅ NEW: Velocity-based classification for fast movements
            elif abs(elbow_velocity) > 5:  # Fast movement detected
                # Use velocity direction to predict phase
                if elbow_velocity < 0:  # Angle decreasing = going down
                    raw_phase = "down" if is_aligned else "down_bad_form"
                else:  # Angle increasing = going up
                    raw_phase = "up" if is_aligned else "up_bad_form"
            
            # Transition phase
            else:
                raw_phase = "transition"
            
            # ✅ NEW: Apply phase smoothing
            self.raw_phase_history.append(raw_phase)
            smoothed_phase = self._smooth_phase(raw_phase)
            
            return smoothed_phase, phase_data
                
        except (IndexError, TypeError) as e:
            return "error", {'error': str(e)}
        

    
    def _smooth_phase(self, raw_phase: str) -> str:
        """✅ NEW: Smooth phase detection to handle velocity changes"""
        if len(self.raw_phase_history) < 3:
            return raw_phase
        
        # Get last 3 phases
        recent = list(self.raw_phase_history)[-3:]
        
        # ✅ If we're in transition but surrounded by same phase, use that phase
        # This handles: down -> transition -> down (fast pushup)
        if recent[-1] == "transition" and len(set(recent[:2])) == 1:
            # Both previous phases are the same
            dominant_phase = recent[0]
            # If they're both down or both up, use that instead of transition
            if 'down' in dominant_phase or 'up' in dominant_phase:
                return dominant_phase
        
        # ✅ Use majority voting for stability
        phase_counts = {}
        for phase in recent:
            base_phase = phase.replace('_bad_form', '')  # Normalize
            phase_counts[base_phase] = phase_counts.get(base_phase, 0) + 1
        
        # Get most common phase
        dominant_base = max(phase_counts, key=phase_counts.get)
        
        # ✅ If transition is brief (surrounded by same phase), ignore it
        if dominant_base != "transition" and phase_counts[dominant_base] >= 2:
            # Restore form suffix if needed
            if '_bad_form' in raw_phase:
                return f"{dominant_base}_bad_form"
            return dominant_base
        
        return raw_phase

    
    def update(self, landmarks: List[Tuple[float, float]]) -> dict:
        """Update detector with new frame"""
        current_time = time.time()
        
        # Detect current phase
        phase, phase_data = self.detect_pushup_phase(landmarks)
        
        # Update phase tracking
        if phase != self.current_phase:
            self.last_phase = self.current_phase
            self.current_phase = phase
            self.phase_start_time = current_time
        
        self.time_in_phase = current_time - self.phase_start_time
        self.phase_history.append(phase)
        
        # Track form issues
        if not phase_data.get('is_aligned', True):
            self.form_issues = phase_data.get('alignment_issues', [])
        else:
            self.form_issues = []
        
        # Check for rep completion
        rep_completed = self.check_rep_completion()
        
        if rep_completed and (current_time - self.last_rep_time) > self.min_rep_interval:
            self.count_rep()
            self.last_rep_time = current_time
            
            if len(self.form_issues) == 0:
                self.good_form_count += 1
        
        form_percentage = (self.good_form_count / self.total_reps * 100) if self.total_reps > 0 else 100
        
        return {
            'phase': self.current_phase,
            'rep_count': self.rep_count,
            'rep_completed': rep_completed,
            'form_issues': self.form_issues,
            'form_percentage': form_percentage,
            'phase_data': phase_data,
            'calibrated': self.is_calibrated
        }
    
    def check_rep_completion(self) -> bool:
        """Count ALL reps (good and bad form)"""
        if len(self.phase_history) < 3:
            return False

        recent_phases = list(self.phase_history)[-4:]

        # ✅ UPDATED: Include bad form phases in patterns
        valid_patterns = [
            # Good form patterns
            ['down', 'transition', 'up', 'up'],
            ['down', 'down', 'up', 'up'],
            ['down', 'up', 'up', 'up'],
            
            # Bad form patterns (still counts as reps!)
            ['down_bad_form', 'transition', 'up', 'up'],
            ['down_bad_form', 'down_bad_form', 'up', 'up'],
            ['down_bad_form', 'up', 'up', 'up'],
            ['down', 'transition', 'up_bad_form', 'up_bad_form'],
            ['down', 'down', 'up_bad_form', 'up_bad_form'],
            ['down', 'up_bad_form', 'up_bad_form', 'up_bad_form'],
            ['down_bad_form', 'transition', 'up_bad_form', 'up_bad_form'],
            ['down_bad_form', 'down_bad_form', 'up_bad_form', 'up_bad_form'],
            ['down_bad_form', 'up_bad_form', 'up_bad_form', 'up_bad_form'],

            # Mixed patterns (some good, some bad form)
            ['down', 'transition', 'up_bad_form', 'up'],
            ['down_bad_form', 'transition', 'up', 'up_bad_form'],
            ['down', 'down_bad_form', 'up', 'up'],
            ['down_bad_form', 'down', 'up_bad_form', 'up'],
        ]
        
        for pattern in valid_patterns:
            if recent_phases == pattern:
                return True
        
        return False
    
    def count_rep(self):
        """Increment rep counter"""
        self.rep_count += 1
        self.total_reps += 1
        print(f"✅ Rep #{self.rep_count} completed!")
        
        if len(self.form_issues) > 0:
            print(f"⚠️ Form issues: {', '.join(self.form_issues)}")
    
    def get_feedback(self) -> str:
        """Get real-time form feedback"""
        if self.current_phase == "error":
            return "⚠️ Can't see your full body - step back from camera"
        
        if len(self.form_issues) > 0:
            feedback_map = {
                'hips_sagging': "⬆️ Engage your core - hips are sagging",
                'hips_too_high': "⬇️ Lower your hips - keep body straight",
                'body_not_straight': "📏 Keep your body in a straight line",
                'landmark_error': "⚠️ Make sure your full body is visible"
            }
            return feedback_map.get(self.form_issues[0], "⚠️ Check your form")
        
        if self.current_phase == "down":
            return "💪 Good! Now push up"
        elif self.current_phase == "up":
            return "✅ Perfect! Lower down"
        elif self.current_phase == "transition":
            return "🔄 Keep moving..."
        
        return "🎯 Start your push-up"
    
    def reset(self):
        """Reset detector state"""
        self.rep_count = 0
        self.current_phase = "neutral"
        self.last_phase = "neutral"
        self.phase_history.clear()
        self.form_issues = []
        self.good_form_count = 0
        self.total_reps = 0
        print("🔄 Push-up detector reset")
    
    def get_stats(self) -> Dict:
        """Get detailed statistics"""
        form_percentage = (self.good_form_count / self.total_reps * 100) if self.total_reps > 0 else 100
        
        return {
            'total_reps': self.rep_count,
            'good_form_reps': self.good_form_count,
            'form_percentage': form_percentage,
            'current_phase': self.current_phase,
            'calibrated': self.is_calibrated,
            'calibration_data': {
                'shoulder_width': self.calibrator.get_measurement('shoulder_width') if self.is_calibrated else None,
                'torso_length': self.calibrator.get_measurement('torso_length') if self.is_calibrated else None,
            } if self.is_calibrated else None
        }