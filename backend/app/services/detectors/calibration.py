import asyncio
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from collections import deque
import math
from datetime import datetime
from uuid import uuid4

class BodyCalibrator:
    """Universal body calibration system for all exercise detectors"""
    
    def __init__(self, user_id: str = None, firebase_client=None):
        self.user_id = user_id
        self.firebase_client = firebase_client
        self.calibration_frames = []
        self.baseline_measurements = None
        self.is_calibrated = False
        self.calibration_target_frames = 120 
        self.calibration_id = None

        # ✅ RELAXED VALIDATION THRESHOLDS - Much more lenient
        self.FRONT_SHOULDER_LEVEL_TOLERANCE = 0.15      # Very lenient shoulder level
        self.FRONT_HIP_LEVEL_TOLERANCE = 0.15           # Very lenient hip level
        self.FRONT_ARM_OUT_DISTANCE = 0.0               # No arm spreading required
        
        self.SIDE_MIN_HEIGHT_RATIO = 0.2                # Very lenient height requirement
        self.SIDE_VERTICAL_TOLERANCE = 0.3              # Very lenient alignment
        
        # Frame requirements
        self.FRONT_FRAME_COUNT = 60                     
        self.SIDE_FRAME_COUNT = 60                     
        
    def calculate_distance(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """Calculate distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_angle(self, point1: Tuple[float, float], 
                       point2: Tuple[float, float], 
                       point3: Tuple[float, float]) -> float:
        """Calculate angle between three points"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    async def add_calibration_frame(self, landmarks: List[Tuple[float, float]], 
                                   exercise_type: str = "general") -> Dict:
        """Add frame to calibration data"""
        if len(landmarks) != 33:
            return {
                'status': 'error',
                'message': 'Invalid pose landmarks',
                'progress': len(self.calibration_frames),
                'target': self.calibration_target_frames
            }
        
        # ✅ SIMPLIFIED VALIDATION - Much more lenient
        if self._is_pose_visible(landmarks):
            self.calibration_frames.append(landmarks)
            print(f"✅ Frame {len(self.calibration_frames)} collected")  # Debug
        else:
            print("❌ Pose not visible enough")  # Debug
        
        progress = len(self.calibration_frames)
        
        if progress >= self.calibration_target_frames:
            self._compute_baseline_measurements()
            
            # Auto-save to Firebase if configured
            save_success = False
            if self.user_id and self.firebase_client:
                save_success = await self.save_calibration_to_firebase()
            
            return {
                'status': 'complete',
                'message': 'Calibration complete! Ready to start workout.',
                'progress': progress,
                'target': self.calibration_target_frames,
                'measurements': self.baseline_measurements,
                'saved_to_cloud': save_success
            }
        
        return {
            'status': 'collecting',
            'message': self._get_calibration_instruction(exercise_type, progress),
            'progress': progress,
            'target': self.calibration_target_frames
        }
    
    def _get_calibration_instruction(self, exercise_type: str, progress: int) -> str:
        """Simplified universal calibration instructions"""
        remaining = self.calibration_target_frames - progress
        
        if progress == 0:
            return "📏 Quick Body Measurement (30 seconds total)\n\nStep 1: Front View - Stand naturally, full body visible"
        elif progress < self.calibration_target_frames // 2:
            return f"📷 Front view... hold steady ({self.calibration_target_frames // 2 - progress} frames remaining)"
        elif progress == self.calibration_target_frames // 2:
            return "Step 2: Side View - Turn 90° left or right, stand naturally"
        else:
            return f"📷 Side view... almost done! ({remaining} frames remaining)"
    
    def _is_pose_visible(self, landmarks: List[Tuple[float, float]]) -> bool:
        """✅ SUPER SIMPLE VALIDATION - Just check key points are visible"""
        try:
            # Key points we need for measurements
            key_points = [
                landmarks[0],   # nose
                landmarks[11],  # left shoulder
                landmarks[12],  # right shoulder
                landmarks[23],  # left hip
                landmarks[24],  # right hip
                landmarks[27],  # left ankle
                landmarks[28],  # right ankle
            ]
            
            # Just check they're all detected (not None)
            visible_count = sum(1 for point in key_points if point is not None)
            
            # Need at least 5 out of 7 key points visible
            is_visible = visible_count >= 5
            
            if not is_visible:
                print(f"Only {visible_count}/7 key points visible")
            
            return is_visible
            
        except (IndexError, TypeError) as e:
            print(f"Error checking pose visibility: {e}")
            return False
    
    def _compute_baseline_measurements(self):
        """Compute measurements from collected frames"""
        
        print(f"📊 Computing measurements from {len(self.calibration_frames)} frames")
        
        # Split frames based on actual collection
        midpoint = len(self.calibration_frames) // 2
        front_frames = self.calibration_frames[:midpoint]
        side_frames = self.calibration_frames[midpoint:]
        
        print(f"📐 Front frames: {len(front_frames)}, Side frames: {len(side_frames)}")

        # Compute measurements
        front_measurements = self._analyze_front_frames(front_frames)
        side_measurements = self._analyze_side_frames(side_frames)
        
        # Combine measurements
        self.baseline_measurements = {
            **front_measurements,
            **side_measurements,
            'calibration_type': 'universal_poses',
            'total_frames': len(self.calibration_frames),
            'created_at': datetime.utcnow().isoformat()
        }
        
        self.is_calibrated = True
        self.calibration_id = self._generate_calibration_id()
        self.calibration_frames = []  # Clear to save memory
        
        print(f"✅ Calibration complete with ID: {self.calibration_id}")

    def _analyze_front_frames(self, frames):
        """Extract measurements from front-facing frames"""
        measurements = {
            'shoulder_width': [],
            'hip_width': [], 
            'arm_span': [],
        }
        
        for landmarks in frames:
            try:
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12] 
                left_hip = landmarks[23]
                right_hip = landmarks[24]
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]
                
                # Only add measurements if points exist
                if left_shoulder and right_shoulder:
                    measurements['shoulder_width'].append(
                        self.calculate_distance(left_shoulder, right_shoulder)
                    )
                
                if left_hip and right_hip:
                    measurements['hip_width'].append(
                        self.calculate_distance(left_hip, right_hip)
                    )
                
                if left_wrist and right_wrist:
                    measurements['arm_span'].append(
                        self.calculate_distance(left_wrist, right_wrist)
                    )
            except (IndexError, TypeError):
                continue
        
        # Compute averages with fallbacks
        shoulder_width_mean = np.mean(measurements['shoulder_width']) if measurements['shoulder_width'] else 0.3
        hip_width_mean = np.mean(measurements['hip_width']) if measurements['hip_width'] else 0.25
        arm_span_mean = np.mean(measurements['arm_span']) if measurements['arm_span'] else 0.5
        
        return {
            'shoulder_width': float(shoulder_width_mean),
            'hip_width': float(hip_width_mean),
            'arm_span': float(arm_span_mean),
            'shoulder_to_hip_ratio': float(shoulder_width_mean / hip_width_mean) if hip_width_mean > 0 else 1.2
        }

    def _analyze_side_frames(self, frames):
        """Extract measurements from side-view frames"""
        measurements = {
            'torso_length': [],
            'leg_length': [],
            'total_height': []
        }
        
        for landmarks in frames:
            try:
                # Use best available side landmarks
                shoulder = landmarks[11] if landmarks[11] else landmarks[12]
                hip = landmarks[23] if landmarks[23] else landmarks[24]
                knee = landmarks[25] if landmarks[25] else landmarks[26]
                ankle = landmarks[27] if landmarks[27] else landmarks[28]
                nose = landmarks[0]
                
                if shoulder and hip:
                    measurements['torso_length'].append(
                        self.calculate_distance(shoulder, hip)
                    )
                
                if hip and ankle:
                    measurements['leg_length'].append(
                        self.calculate_distance(hip, ankle)
                    )
                
                if nose and ankle:
                    measurements['total_height'].append(
                        self.calculate_distance(nose, ankle)
                    )
            except (IndexError, TypeError):
                continue
        
        # Compute averages with fallbacks
        torso_length_mean = np.mean(measurements['torso_length']) if measurements['torso_length'] else 0.3
        leg_length_mean = np.mean(measurements['leg_length']) if measurements['leg_length'] else 0.4
        total_height_mean = np.mean(measurements['total_height']) if measurements['total_height'] else 0.7
        
        return {
            'torso_length': float(torso_length_mean),
            'leg_length': float(leg_length_mean),
            'total_height': float(total_height_mean),
            'leg_to_torso_ratio': float(leg_length_mean / torso_length_mean) if torso_length_mean > 0 else 1.3
        }
    
    def get_normalized_distance(self, distance: float, reference: str = 'shoulder_width') -> float:
        """Normalize distance using baseline measurements"""
        if not self.is_calibrated:
            return distance
        
        reference_value = self.baseline_measurements.get(reference, 1.0)
        return distance / reference_value if reference_value > 0 else distance
    
    def get_measurement(self, key: str) -> Optional[float]:
        """Get specific measurement"""
        if not self.is_calibrated:
            return None
        return self.baseline_measurements.get(key)
    
    def reset_calibration(self):
        """Reset calibration data"""
        self.calibration_frames = []
        self.baseline_measurements = None
        self.is_calibrated = False
        self.calibration_id = None
    
    def _generate_calibration_id(self) -> str:
        """Generate unique calibration ID"""
        return f"cal_{int(datetime.utcnow().timestamp())}_{str(uuid4())[:4]}"
    
    async def save_calibration_to_firebase(self) -> bool:
        """Save calibration to Firebase"""
        if not self.user_id or not self.firebase_client or not self.baseline_measurements:
            return False
            
        try:
            calibration_data = {
                'measurements': self.baseline_measurements,
                'is_calibrated': self.is_calibrated,
                'calibration_id': self.calibration_id or self._generate_calibration_id(),
                'created_at': datetime.utcnow().isoformat(),
                'version': '1.0'
            }
            
            # ✅ FIXED: Save to users collection (consistent)
            user_ref = self.firebase_client.collection('users').document(self.user_id)
            await user_ref.update({
                'calibrationData': calibration_data
            })
            
            print(f"✅ Calibration saved to Firebase for user {self.user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving calibration: {e}")
            return False
    
    def load_user_calibration(self) -> bool:
        """Load existing calibration from Firebase (synchronous)"""
        if not self.user_id or not self.firebase_client:
            print("⚠️ Missing user_id or firebase_client")
            return False
            
        try:
            # ✅ FIXED: Load from users collection (consistent with save)
            user_ref = self.firebase_client.collection('users').document(self.user_id)
            doc = user_ref.get()
            
            if doc.exists:
                user_data = doc.to_dict()
                calibration_data = user_data.get('calibrationData')
                
                if calibration_data and isinstance(calibration_data, dict):
                    measurements = calibration_data.get('measurements')
                    is_calibrated = calibration_data.get('is_calibrated', False)
                    calibration_id = calibration_data.get('calibration_id')
                    
                    # Validate measurements exist and are valid
                    if measurements and isinstance(measurements, dict) and is_calibrated:
                        self.baseline_measurements = measurements
                        self.is_calibrated = is_calibrated
                        self.calibration_id = calibration_id
                        
                        print(f"✅ Loaded existing calibration for user {self.user_id}")
                        print(f"📊 Measurements: {list(measurements.keys())}")
                        return True
                    else:
                        print("⚠️ Calibration data exists but is incomplete")
                        return False
                else:
                    print("📝 No calibration data found in user document")
                    return False
            else:
                print("❌ User document not found")
                return False
                    
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            import traceback
            traceback.print_exc()
            
        return False

async def run_calibration_session(calibrator, get_landmarks_callback, exercise_type: str = "general"):
    """Run a full calibration session with phase tracking and detailed progress updates"""
    try:
        calibrator.reset_calibration()
        print("🚀 Starting calibration session...")

        current_phase = "front"
        frames_collected = 0
        phase_frames = calibrator.calibration_target_frames // 2

        while frames_collected < calibrator.calibration_target_frames:
            landmarks = await get_landmarks_callback()
            
            if not landmarks:
                print("❌ No landmarks detected, skipping frame")
                await asyncio.sleep(0.1)
                continue

            result = await calibrator.add_calibration_frame(landmarks, exercise_type)
            frames_collected = result['progress']

            # Phase transition detection
            if frames_collected == phase_frames:
                current_phase = "side"
                print("\n🔄 Transition: Please turn to side view\n")
                # Give user time to turn
                await asyncio.sleep(3)

            # Calculate phase-specific progress
            phase_progress = (
                frames_collected if current_phase == "front" 
                else frames_collected - phase_frames
            )
            total_progress = (frames_collected / calibrator.calibration_target_frames) * 100

            # Detailed progress update
            print(
                f"📊 Progress: {frames_collected}/{calibrator.calibration_target_frames} "
                f"({total_progress:.1f}%) - Phase: {current_phase.upper()} "
                f"({phase_progress}/{phase_frames})\n"
                f"ℹ️  {result['message']}"
            )

            if result['status'] == 'error':
                print(f"⚠️ Frame warning: {result['message']}")

            await asyncio.sleep(0.1)  # Frame delay

        # Final measurements and storage
        if calibrator.baseline_measurements:
            print("\n📏 Final Measurements:")
            for key, value in calibrator.baseline_measurements.items():
                if isinstance(value, float):
                    print(f"  • {key}: {value:.3f}")

        print("\n🎉 Calibration session complete!")
        return {
            "status": "success",
            "frames_collected": frames_collected,
            "measurements": calibrator.baseline_measurements
        }

    except asyncio.CancelledError:
        print("\n⚠️ Calibration session cancelled")
        calibrator.reset_calibration()
        raise

    except Exception as e:
        print(f"\n❌ Calibration error: {str(e)}")
        calibrator.reset_calibration()
        raise