from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import joblib
from datetime import datetime
from pathlib import Path
from app.services.mediapipe.base_detector import BasePoseDetector

# Load the same models as realtime_detect.py
print('🔧 Loading hybrid system for WebSocket...')

# Load Phase 1: Exercise Detection
try:
    exercise_model = joblib.load('app/models/exercise_detector.pkl')
    exercise_encoder = joblib.load('app/models/exercise_detector_encoder.pkl')
    print('✅ Exercise detector loaded')
except Exception as e:
    print(f'❌ Error loading exercise detector: {e}')

# Load Phase 2: Form Coaches
form_coaches = {}
form_encoders = {}

for exercise in ['squat', 'plank', 'pushup']:
    try:
        form_coaches[exercise] = joblib.load(f'app/models/{exercise}_form_coach.pkl')
        form_encoders[exercise] = joblib.load(f'app/models/{exercise}_form_coach_encoder.pkl')
        print(f'✅ {exercise} form coach loaded')
    except Exception as e:
        print(f'❌ {exercise} form coach not found: {e}')

# Load scaler
try:
    scaler_files = list(Path('ml/data/processed').glob('exercise_detection_scaler_*.pkl'))
    if scaler_files:
        scaler = joblib.load(max(scaler_files, key=lambda x: x.stat().st_mtime))
        print('✅ Scaler loaded')
    else:
        print('❌ No scaler found')
        scaler = None
except Exception as e:
    print(f'❌ Error loading scaler: {e}')
    scaler = None

pose_detector = BasePoseDetector(static_image_mode=False, min_detection_confidence=0.7)
router = APIRouter()

def extract_features(coords):
    """Extract features from pose coordinates - same as realtime_detect.py"""
    coords_array = np.array(coords)
    if coords_array.shape != (33, 2):
        return None
    
    features = []
    
    # Normalize pose
    shoulder_width = np.linalg.norm(coords_array[11] - coords_array[12])
    if shoulder_width > 0:
        hip_center = (coords_array[23] + coords_array[24]) / 2
        normalized_coords = (coords_array - hip_center) / shoulder_width
    else:
        return None
    
    # Key landmarks
    key_landmarks = {
        'nose': 0, 'left_eye': 2, 'right_eye': 5,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28
    }
    
    for idx in key_landmarks.values():
        features.extend(normalized_coords[idx])
    
    # Engineered features
    try:
        shoulder_center = (coords_array[11] + coords_array[12]) / 2
        torso_vector = shoulder_center - hip_center
        torso_angle = np.arctan2(torso_vector[1], torso_vector[0]) * 180 / np.pi
        features.append(torso_angle)
        
        head_to_hip = np.linalg.norm(coords_array[0] - hip_center)
        compactness = head_to_hip / shoulder_width
        features.append(compactness)
        
        def calc_angle(p1, p2, p3):
            v1, v2 = p1 - p2, p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        left_arm_angle = calc_angle(coords_array[11], coords_array[13], coords_array[15])
        right_arm_angle = calc_angle(coords_array[12], coords_array[14], coords_array[16])
        left_leg_angle = calc_angle(coords_array[23], coords_array[25], coords_array[27])
        right_leg_angle = calc_angle(coords_array[24], coords_array[26], coords_array[28])
        
        features.extend([left_arm_angle, right_arm_angle, left_leg_angle, right_leg_angle])
    except:
        features.extend([0.0] * 6)
    
    while len(features) < 36:
        features.append(0.0)
    return np.array(features[:36])

def analyze_hybrid(coords):
    """Complete hybrid analysis: exercise detection + form coaching - same as realtime_detect.py"""
    try:
        features = extract_features(coords)
        if features is None or scaler is None:
            return 'unknown', 0.0, None, 0.0
        
        # Phase 1: Exercise Detection
        features_2d = features.reshape(1, -1)
        features_2d = scaler.transform(features_2d)
        
        exercise_pred = exercise_model.predict(features_2d)[0]
        exercise_probs = exercise_model.predict_proba(features_2d)[0]
        exercise = exercise_encoder.inverse_transform([exercise_pred])[0]
        exercise_confidence = max(exercise_probs)
        
        # Phase 2: Form Analysis (if we have a coach for this exercise)
        form_quality = None
        form_confidence = 0.0
        
        if exercise in form_coaches and exercise_confidence > 0.6:  # Raised back to 0.6 for more accurate form detection
            form_model = form_coaches[exercise]
            form_encoder = form_encoders[exercise]
            
            # Use same features for form analysis
            form_pred = form_model.predict(features_2d)[0]
            form_probs = form_model.predict_proba(features_2d)[0]
            form_quality = form_encoder.inverse_transform([form_pred])[0]
            form_confidence = max(form_probs)
            
            # Be more lenient with form quality for beginners
            # If confidence is low, default to "good" instead of "poor"
            if form_confidence < 0.6 and form_quality == 'poor':
                form_quality = 'good'
                form_confidence = 0.6
        
        return exercise, exercise_confidence, form_quality, form_confidence
        
    except Exception as e:
        print(f'Analysis error: {e}')
        return 'unknown', 0.0, None, 0.0

class ImageInput(BaseModel):
    image: str  # base64 string

@router.websocket("/pose_detection")
async def stream_pose_estimation(ws: WebSocket, exercise: str = "general"):
    await ws.accept()
    print(f"✅ WebSocket connection accepted for {exercise} exercise")
    frame_count = 0
    detection_history = []  # Track recent detections for stability
    
    # Exercise filtering - only detect specified exercise or general detection
    valid_exercises = ['squat', 'pushup', 'plank', 'general']
    if exercise not in valid_exercises:
        exercise = 'general'
        
    print(f"🎯 Exercise mode: {exercise}")
    
    try:
        while True:
            data = await ws.receive_text()  # base64 string
            frame_count += 1
            
            # Process every frame but log less frequently
            if frame_count % 10 == 0:
                print(f"📸 Processed {frame_count} frames for {exercise}")
            
            if not data:
                continue
                
            try:
                img_data = base64.b64decode(data)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame_count % 20 == 0:  # Log every 20th frame
                    print(f"🖼️ Frame {frame_count} decoded: {frame.shape if frame is not None else 'None'}")
                    
            except Exception as decode_error:
                print(f"❌ Image decode error on frame {frame_count}: {decode_error}")
                await ws.send_json({
                    "error": "decode_error",
                    "message": f"Could not decode image: {str(decode_error)}"
                })
                continue
            
            if frame is None:
                await ws.send_json({
                    "error": "invalid_frame",
                    "message": "Could not decode frame"
                })
                continue
            
            # Resize frame for faster processing
            height, width = frame.shape[:2]
            if width > 640:  # Resize if too large
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            landmarks = pose_detector.process(frame)
            
            if not landmarks:
                await ws.send_json({
                    "exercise": "none",
                    "exercise_confidence": 0.0,
                    "form_quality": None,
                    "form_confidence": 0.0,
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Extract coordinates and run hybrid analysis
            coords = [(p.x, p.y) for p in landmarks.landmark]
            detected_exercise, exercise_confidence, form_quality, form_confidence = analyze_hybrid(coords)
            
            # Add to detection history for stability
            detection_history.append({
                'exercise': detected_exercise,
                'confidence': exercise_confidence,
                'form': form_quality,
                'form_confidence': form_confidence
            })
            
            # Keep only last 5 detections for moving average
            detection_history = detection_history[-5:]
            
            # Calculate stable detection (majority vote + confidence filtering)
            if len(detection_history) >= 3:
                # Filter high-confidence detections
                reliable_detections = [d for d in detection_history if d['confidence'] > 0.5]
                
                if reliable_detections:
                    # Use most recent reliable detection
                    stable_detection = reliable_detections[-1]
                    detected_exercise = stable_detection['exercise']
                    exercise_confidence = stable_detection['confidence']
                    form_quality = stable_detection['form']
                    form_confidence = stable_detection['form_confidence']
                    
                    # Smooth confidence values (moving average)
                    recent_confidences = [d['confidence'] for d in detection_history[-3:] if d['exercise'] == detected_exercise]
                    if recent_confidences:
                        exercise_confidence = sum(recent_confidences) / len(recent_confidences)
            
            # Exercise filtering logic
            if exercise != 'general':
                # Stricter filtering to prevent false positives
                if detected_exercise != exercise or exercise_confidence < 0.6:
                    await ws.send_json({
                        "exercise": "none",
                        "exercise_confidence": 0.0,
                        "form_quality": None,
                        "form_confidence": 0.0,
                        "feedback": {
                            "message": f"Position yourself to perform {exercise}s",
                            "type": "info",
                            "corrections": []
                        },
                        "timestamp": datetime.now().isoformat()
                    })
                    continue
                    
                # High confidence threshold for exercise detection to prevent false positives
                if exercise_confidence < 0.7:  # Raised back to 0.7 for stability
                    await ws.send_json({
                        "exercise": detected_exercise,
                        "exercise_confidence": float(exercise_confidence),
                        "form_quality": None,
                        "form_confidence": 0.0,
                        "feedback": {
                            "message": f"Position yourself clearly for {exercise} detection",
                            "type": "info",
                            "corrections": []
                        },
                        "timestamp": datetime.now().isoformat()
                    })
                    continue
            
            if frame_count % 10 == 0:  # Log every 10th analysis
                print(f"🏋️ Analysis {frame_count}: {detected_exercise} ({exercise_confidence:.2f}) | Form: {form_quality} ({form_confidence:.2f})")
            
            # Generate feedback based on results
            feedback = generate_feedback(detected_exercise, form_quality, form_confidence)
            
            # Send comprehensive results
            response = {
                "exercise": detected_exercise,
                "exercise_confidence": float(exercise_confidence),
                "form_quality": form_quality,
                "form_confidence": float(form_confidence) if form_confidence else 0.0,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            }
            await ws.send_json(response)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected after {frame_count} frames")
    except Exception as e:
        print(f"WebSocket error after {frame_count} frames: {e}")
        import traceback
        traceback.print_exc()
        try:
            await ws.send_json({
                "error": "processing_error", 
                "message": str(e)
            })
        except:
            pass
        await ws.close()

def generate_feedback(exercise: str, form_quality: str, form_confidence: float):
    """Generate coaching feedback based on analysis results"""
    if not form_quality or form_confidence < 0.3:  # Lowered from 0.5 to 0.3 for beginners
        return {
            "message": "Analyzing your form...",
            "type": "analyzing",
            "corrections": []
        }
    
    feedback_map = {
        "excellent": {
            "message": "Perfect form! 🏆 Keep it up!",
            "type": "success",
            "corrections": []
        },
        "good": {
            "message": "Good form! Minor adjustments needed",
            "type": "warning", 
            "corrections": get_exercise_tips(exercise, "good")
        },
        "poor": {
            "message": "Focus on your technique",
            "type": "error",
            "corrections": get_exercise_tips(exercise, "poor")
        }
    }
    
    return feedback_map.get(form_quality, {
        "message": "Keep working on your form",
        "type": "info",
        "corrections": []
    })

def get_exercise_tips(exercise: str, quality: str):
    """Exercise-specific coaching tips"""
    tips = {
        "squat": {
            "good": ["Keep knees aligned with toes", "Maintain straight back"],
            "poor": ["Lower down further", "Don't let knees cave in", "Keep chest up"]
        },
        "pushup": {
            "good": ["Maintain straight line from head to heels", "Control the descent"],
            "poor": ["Don't let hips sag", "Go down to chest level", "Keep core tight"]
        },
        "plank": {
            "good": ["Hold the position steady", "Keep breathing"],
            "poor": ["Don't lift hips too high", "Engage your core", "Keep straight line"]
        }
    }
    
    return tips.get(exercise, {}).get(quality, ["Focus on proper technique"])