# from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
# from pydantic import BaseModel
# import base64
# import numpy as np
# import cv2
# import joblib
# import json
# from datetime import datetime
# from pathlib import Path
# from app.services.mediapipe.base_detector import BasePoseDetector
# from app.services.movement_detector import MovementBasedDetector
# import warnings
# from collections import deque
# import time

# # Suppress the specific sklearn warning about feature names
# warnings.filterwarnings("ignore", message="X does not have valid feature names")
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# # Load the same models as realtime_detect.py (OPTIONAL - MovementBasedDetector works without them)
# print('🔧 Loading hybrid system for WebSocket...')
# print('ℹ️  Note: ML models are optional - MovementBasedDetector uses optimized rule-based detection')

# # Load Phase 1: Exercise Detection (OPTIONAL)
# exercise_model = None
# exercise_encoder = None
# try:
#     exercise_model = joblib.load('app/models/exercise_detector.pkl')
#     exercise_encoder = joblib.load('app/models/exercise_detector_encoder.pkl')
#     print('✅ Exercise detector loaded')
# except Exception as e:
#     print(f'ℹ️  Exercise detector not available: {e}')
#     print('✅ Using MovementBasedDetector rule-based exercise detection instead')

# # Load Phase 2: Form Coaches (OPTIONAL)
# form_coaches = {}
# form_encoders = {}

# for exercise in ['squat', 'plank', 'pushup']:
#     try:
#         form_coaches[exercise] = joblib.load(f'app/models/{exercise}_form_coach.pkl')
#         form_encoders[exercise] = joblib.load(f'app/models/{exercise}_form_coach_encoder.pkl')
#         print(f'✅ {exercise} form coach loaded')
#     except Exception as e:
#         print(f'ℹ️  {exercise} form coach not available: {e}')

# # Summary of loaded form coaches
# if form_coaches:
#     print(f"📋 ML Form coaches loaded: {list(form_coaches.keys())}")
#     print(f"📋 ML Form encoders loaded: {list(form_encoders.keys())}")
# else:
#     print("ℹ️  No ML form coaches available - using rule-based form analysis")

# # Load scaler (OPTIONAL)
# scaler = None
# try:
#     scaler_files = list(Path('ml/data/processed').glob('exercise_detection_scaler_*.pkl'))
#     if scaler_files:
#         scaler = joblib.load(max(scaler_files, key=lambda x: x.stat().st_mtime))
#         print('✅ Scaler loaded')
#     else:
#         print('ℹ️  No scaler found - using direct feature analysis')
#         scaler = None
# except Exception as e:
#     print(f'ℹ️  Scaler not available: {e}')
#     scaler = None

# print('🚀 MovementBasedDetector system ready - optimized rule-based detection active!')

# pose_detector = BasePoseDetector(static_image_mode=False, min_detection_confidence=0.7)
# router = APIRouter()

# class MovementRepDetector:
#     """Wrapper around MovementBasedDetector for backward compatibility"""
#     def __init__(self, exercise_type='pushup'):
#         self.exercise_type = exercise_type
#         self.movement_detector = MovementBasedDetector()
#         self.rep_count = 0
        
#         # Set the current exercise in the movement detector
#         self.movement_detector.current_exercise = exercise_type
#         # Reduced logging for performance
    
#     def extract_movement_features(self, coords):
#         """Convert coords format for MovementBasedDetector"""
#         # Convert (x, y) tuples to flat list format expected by MovementBasedDetector
#         if len(coords) == 33 and all(len(coord) == 2 for coord in coords):
#             landmark_list = []
#             for x, y in coords:
#                 landmark_list.extend([x, y, 0.0])  # Add z=0 since we only have 2D coords
#             return landmark_list
#         return None
    
#     def detect_rep_completion(self, movement_data):
#         """Process frame using MovementBasedDetector"""
#         if movement_data is None:
#             return False
        
#         # Process the frame with MovementBasedDetector
#         result = self.movement_detector.process_frame(movement_data, self.exercise_type)
        
#         # Check if a rep was completed
#         rep_completed = result.get('rep_completed', False)
#         if rep_completed:
#             self.rep_count = result.get('total_reps', self.rep_count)
#             return True
        
#         # Update rep count from movement detector
#         self.rep_count = result.get('total_reps', self.rep_count)
#         return False
    
#     def count_rep(self):
#         """Get current rep count with encouraging message"""
#         encouraging_messages = [
#             f"🎉 Rep {self.rep_count} completed! Keep it up!",
#             f"💪 That's {self.rep_count} reps! You're getting stronger!",
#             f"🔥 Awesome! {self.rep_count} reps down!",
#             f"⭐ Great form on rep {self.rep_count}!",
#             f"🚀 {self.rep_count} reps! You're crushing it!"
#         ]
#         if self.rep_count > 0:
#             # Reduced print overhead for performance
#             pass
#         return self.rep_count

# def extract_features(coords):
#     """Extract features from pose coordinates - same as realtime_detect.py"""
#     coords_array = np.array(coords)
#     if coords_array.shape != (33, 2):
#         return None
    
#     features = []
    
#     # Normalize pose
#     shoulder_width = np.linalg.norm(coords_array[11] - coords_array[12])
#     if shoulder_width > 0:
#         hip_center = (coords_array[23] + coords_array[24]) / 2
#         normalized_coords = (coords_array - hip_center) / shoulder_width
#     else:
#         return None
    
#     # Key landmarks
#     key_landmarks = {
#         'nose': 0, 'left_eye': 2, 'right_eye': 5,
#         'left_shoulder': 11, 'right_shoulder': 12,
#         'left_elbow': 13, 'right_elbow': 14,
#         'left_wrist': 15, 'right_wrist': 16,
#         'left_hip': 23, 'right_hip': 24,
#         'left_knee': 25, 'right_knee': 26,
#         'left_ankle': 27, 'right_ankle': 28
#     }
    
#     for idx in key_landmarks.values():
#         features.extend(normalized_coords[idx])
    
#     # Engineered features
#     try:
#         shoulder_center = (coords_array[11] + coords_array[12]) / 2
#         torso_vector = shoulder_center - hip_center
#         torso_angle = np.arctan2(torso_vector[1], torso_vector[0]) * 180 / np.pi
#         features.append(torso_angle)
        
#         head_to_hip = np.linalg.norm(coords_array[0] - hip_center)
#         compactness = head_to_hip / shoulder_width
#         features.append(compactness)
        
#         def calc_angle(p1, p2, p3):
#             v1, v2 = p1 - p2, p3 - p2
#             cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#             return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
#         left_arm_angle = calc_angle(coords_array[11], coords_array[13], coords_array[15])
#         right_arm_angle = calc_angle(coords_array[12], coords_array[14], coords_array[16])
#         left_leg_angle = calc_angle(coords_array[23], coords_array[25], coords_array[27])
#         right_leg_angle = calc_angle(coords_array[24], coords_array[26], coords_array[28])
        
#         features.extend([left_arm_angle, right_arm_angle, left_leg_angle, right_leg_angle])
#     except:
#         features.extend([0.0] * 6)
    
#     while len(features) < 36:
#         features.append(0.0)
    
#     # Return as numpy array and suppress any sklearn warnings during inference
#     return np.array(features[:36], dtype=np.float64)

# def analyze_hybrid(coords, selected_exercise='pushup'):
#     """Complete hybrid analysis: exercise detection + form coaching - optimized for MovementBasedDetector"""
#     try:
#         # For MovementBasedDetector, we know the exercise from URL parameter
#         # This provides much more reliable exercise detection than ML models
#         detected_exercise = selected_exercise
#         exercise_confidence = 1.0  # 100% confidence since user selected the exercise
        
#         # Phase 2: Form Analysis (if we have ML models available)
#         form_quality = None
#         form_confidence = 0.0
        
#         if exercise_model and scaler and selected_exercise == 'general':
#             # Only use ML exercise detection for general mode
#             features = extract_features(coords)
#             if features is not None:
#                 features_2d = features.reshape(1, -1)
                
#                 # Apply scaling with warning suppression
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")
#                     features_2d = scaler.transform(features_2d)
                
#                 exercise_pred = exercise_model.predict(features_2d)[0]
#                 exercise_probs = exercise_model.predict_proba(features_2d)[0]
#                 detected_exercise = exercise_encoder.inverse_transform([exercise_pred])[0]
#                 exercise_confidence = max(exercise_probs)
                
#                 print(f"🤖 ML Exercise detection: {detected_exercise} ({exercise_confidence:.3f})")
        
#         # Form analysis using ML models if available
#         if detected_exercise in form_coaches and exercise_confidence > 0.8:
#             features = extract_features(coords)
#             if features is not None:
#                 features_2d = features.reshape(1, -1)
                
#                 form_model = form_coaches[detected_exercise]
#                 form_encoder = form_encoders[detected_exercise]
                
#                 # Apply scaling if available
#                 if scaler:
#                     with warnings.catch_warnings():
#                         warnings.simplefilter("ignore")
#                         features_2d = scaler.transform(features_2d)
                
#                 form_pred = form_model.predict(features_2d)[0]
#                 form_probs = form_model.predict_proba(features_2d)[0]
#                 form_quality = form_encoder.inverse_transform([form_pred])[0]
#                 form_confidence = max(form_probs)
                
#                 print(f"🎯 ML Form analysis for {detected_exercise}: quality={form_quality}, confidence={form_confidence:.3f}")
                
#                 # Only accept form analysis if confidence is high enough
#                 if form_confidence < 0.7:
#                     print(f"⚠️ Form confidence too low ({form_confidence:.3f} < 0.7), ignoring ML form analysis")
#                     form_quality = None
#                     form_confidence = 0.0
#         else:
#             if detected_exercise not in form_coaches:
#                 print(f"ℹ️  No ML form coach available for {detected_exercise} - using rule-based analysis")
#             elif exercise_confidence <= 0.8:
#                 print(f"⚠️ Exercise confidence too low for ML form analysis ({exercise_confidence:.3f} <= 0.8)")
        
#         # MovementBasedDetector provides its own form analysis through movement patterns
#         # So we return the exercise info and let MovementBasedDetector handle form feedback
#         return detected_exercise, exercise_confidence, form_quality, form_confidence
        
#     except Exception as e:
#         print(f'Analysis error: {e}')
#         # Return the selected exercise even if ML analysis fails
#         return selected_exercise, 1.0, None, 0.0

# class ImageInput(BaseModel):
#     image: str  # base64 string

# @router.websocket("/pose_detection")
# async def stream_pose_estimation(ws: WebSocket, exercise: str = "general"):
#     await ws.accept()
#     print(f"✅ WebSocket connection accepted for {exercise} exercise")
#     frame_count = 0
#     detection_history = []  # Keep for compatibility but will not be used for performance
#     frontend_ready = False  # Track if frontend is ready to send frames
    
#     # Initialize movement-based rep detector
#     movement_detector = None
#     if exercise in ['pushup', 'squat', 'plank']:
#         movement_detector = MovementRepDetector(exercise)
#         print(f"🏋️ Movement-based rep detector initialized for {exercise}")
    
#     # Exercise filtering - only detect specified exercise or general detection
#     valid_exercises = ['squat', 'pushup', 'plank', 'general']
#     if exercise not in valid_exercises:
#         exercise = 'general'
        
#     print(f"🎯 Exercise mode: {exercise} - Waiting for frontend ready signal")
    
#     # Send initial handshake response
#     await ws.send_json({
#         "type": "connection_established",
#         "exercise_mode": exercise,
#         "message": "Backend ready - send 'frontend_ready' when camera is ready",
#         "timestamp": datetime.now().isoformat()
#     })
    
#     try:
#         while True:
#             data = await ws.receive_text()
            
#             # Handle control messages (non-base64 data)
#             if not frontend_ready:
#                 try:
#                     control_message = json.loads(data)
#                     if control_message.get("type") == "frontend_ready":
#                         frontend_ready = True
#                         print(f"🚀 Frontend ready signal received - starting pose detection for {exercise}")
#                         await ws.send_json({
#                             "type": "backend_ready",
#                             "message": "Backend ready for frame processing",
#                             "timestamp": datetime.now().isoformat()
#                         })
#                         continue
#                     elif control_message.get("type") == "frontend_pause":
#                         frontend_ready = False
#                         print(f"⏸️ Frontend pause signal received")
#                         continue
#                     elif control_message.get("type") == "frontend_resume":
#                         frontend_ready = True
#                         print(f"▶️ Frontend resume signal received")
#                         continue
#                 except json.JSONDecodeError:
#                     # Not a JSON control message, might be base64 frame data
#                     pass
            
#             # Only process frames if frontend is ready
#             if not frontend_ready:
#                 print(f"⏸️ Frontend not ready - skipping frame {frame_count}")
#                 continue
                
#             frame_count += 1
            
#             # Enhanced logging for frame processing debugging
#             if frame_count % 20 == 0:  # Log every 20 frames for debugging
#                 print(f"📸 Frame {frame_count} received - processing for {exercise}")
            
#             if not data:
#                 print(f"❌ Empty data received on frame {frame_count}")
#                 continue
#                 continue
                
#             try:
#                 # Optimized frame decoding pipeline
#                 img_data = base64.b64decode(data)
#                 np_arr = np.frombuffer(img_data, np.uint8)
#                 frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
#                 if frame_count % 200 == 0:  # Even less frequent logging
#                     print(f"🖼️ Frame {frame_count} decoded: {frame.shape if frame is not None else 'None'}")
                    
#             except Exception as decode_error:
#                 print(f"❌ Image decode error on frame {frame_count}: {decode_error}")
#                 await ws.send_json({
#                     "error": "decode_error",
#                     "message": f"Could not decode image: {str(decode_error)}"
#                 })
#                 continue
            
#             if frame is None:
#                 await ws.send_json({
#                     "error": "invalid_frame",
#                     "message": "Could not decode frame"
#                 })
#                 continue
            
#             # Hardware-friendly frame resizing 
#             height, width = frame.shape[:2]
#             if width > 320:  # Back to reasonable size (was 240px - too small)
#                 scale = 320 / width
#                 new_width = int(width * scale)
#                 new_height = int(height * scale)
#                 # Use INTER_LINEAR for good balance of speed and quality
#                 frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
#             try:
#                 landmarks = pose_detector.process(frame)
#             except Exception as pose_error:
#                 print(f"❌ Pose detection error: {pose_error}")
#                 await ws.send_json({
#                     "error": "pose_detection_error",
#                     "message": f"Pose detection failed: {str(pose_error)}"
#                 })
#                 continue
            
#             if not landmarks:
#                 # Skip sending any message for missing landmarks to reduce WebSocket traffic
#                 continue
            
#             # Extract coordinates for immediate processing - NO STABILITY CHECKS for maximum speed
#             coords = [(p.x, p.y) for p in landmarks.landmark]
            
#             # Ultra-fast processing - skip ALL analysis for selected exercises
#             if exercise != 'general':
#                 detected_exercise = exercise
#                 exercise_confidence = 1.0
#                 form_quality = None
#                 form_confidence = 0.0
#             else:
#                 # Only for general mode, run minimal analysis
#                 detected_exercise, exercise_confidence, form_quality, form_confidence = analyze_hybrid(coords, exercise)
            
#             # Movement-based rep counting logic - ULTRA-FAST direct processing
#             rep_completed = False
#             total_reps = 0
            
#             # Direct movement detection without filtering delays
#             if movement_detector:
#                 # Extract movement features for the specific exercise (convert to landmark list format)
#                 movement_data = movement_detector.extract_movement_features(coords)
                
#                 if movement_data:
#                     # Check for rep completion using movement patterns
#                     rep_completed = movement_detector.detect_rep_completion(movement_data)
                    
#                     if rep_completed:
#                         movement_detector.count_rep()  # Just count it internally
                        
#                         # Send simple rep completion signal - frontend will increment
#                         rep_signal = {
#                             "type": "rep_completed",
#                             "exercise": detected_exercise,
#                             "timestamp": datetime.now().isoformat()
#                         }
#                         await ws.send_json(rep_signal)
#                         print(f"🎉 Rep completed signal sent for {detected_exercise}")
                        
#                     else:
#                         total_reps = movement_detector.rep_count
#                 else:
#                     total_reps = movement_detector.rep_count
            
#             # Only send data when absolutely necessary - rep completion signals only for minimal latency
#             # Status updates removed to optimize WebSocket performance
            
#     except WebSocketDisconnect:
#         print(f"WebSocket disconnected after {frame_count} frames")
#     except Exception as e:
#         print(f"WebSocket error after {frame_count} frames: {e}")
#         import traceback
#         traceback.print_exc()
#         try:
#             await ws.send_json({
#                 "error": "processing_error", 
#                 "message": str(e)
#             })
#         except:
#             pass
#         await ws.close()

# def generate_feedback(exercise: str, form_quality: str, form_confidence: float):
#     """Generate coaching feedback based on analysis results"""
#     if not form_quality or form_confidence < 0.3:  # Lowered from 0.5 to 0.3 for beginners
#         return {
#             "message": "Analyzing your form...",
#             "type": "analyzing",
#             "corrections": []
#         }
    
#     feedback_map = {
#         "excellent": {
#             "message": "Perfect form! 🏆 Keep it up!",
#             "type": "success",
#             "corrections": []
#         },
#         "good": {
#             "message": "Great job! 💪 Small tweaks for even better form",
#             "type": "success", 
#             "corrections": get_exercise_tips(exercise, "good")
#         },
#         "poor": {
#             "message": "Nice effort! 👍 Let's improve your technique",
#             "type": "improvement",  # Changed from "error" to be more encouraging
#             "corrections": get_exercise_tips(exercise, "poor")
#         }
#     }
    
#     return feedback_map.get(form_quality, {
#         "message": "Keep working on your form",
#         "type": "info",
#         "corrections": []
#     })

# def get_exercise_tips(exercise: str, quality: str):
#     """Exercise-specific coaching tips - encouraging for beginners"""
#     tips = {
#         "squat": {
#             "good": ["Keep knees aligned with toes", "Maintain straight back", "You're doing great!"],
#             "poor": ["Try lowering down a bit more", "Keep knees pointing forward", "Chest up, you've got this!", "Every rep counts!"]
#         },
#         "pushup": {
#             "good": ["Maintain straight line from head to heels", "Control the descent", "Excellent work!"],
#             "poor": ["Try to keep hips in line", "Lower chest toward ground", "Keep core engaged", "You're building strength!"]
#         },
#         "plank": {
#             "good": ["Hold the position steady", "Keep breathing", "Looking strong!"],
#             "poor": ["Try to keep hips level", "Engage your core muscles", "Hold that line", "Every second counts!"]
#         }
#     }
    
#     return tips.get(exercise, {}).get(quality, ["Keep practicing, you're improving!"])