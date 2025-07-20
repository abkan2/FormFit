import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve()))

import cv2
import numpy as np
import joblib
from app.services.mediapipe.base_detector import BasePoseDetector

print('🔧 Loading hybrid system...')

# Load Phase 1: Exercise Detection
exercise_model = joblib.load('app/models/exercise_detector.pkl')
exercise_encoder = joblib.load('app/models/exercise_detector_encoder.pkl')

# Load Phase 2: Form Coaches
form_coaches = {}
form_encoders = {}

for exercise in ['squat', 'plank', 'pushup']:
    try:
        form_coaches[exercise] = joblib.load(f'app/models/{exercise}_form_coach.pkl')
        form_encoders[exercise] = joblib.load(f'app/models/{exercise}_form_coach_encoder.pkl')
        print(f'✅ {exercise} form coach loaded')
    except:
        print(f'❌ {exercise} form coach not found')

# Load scaler
scaler_files = list(Path('ml/data/processed').glob('exercise_detection_scaler_*.pkl'))
scaler = joblib.load(max(scaler_files, key=lambda x: x.stat().st_mtime))

print(f'✅ Hybrid system ready!')
print(f'Exercise detection: {exercise_encoder.classes_}')

def extract_features(coords):
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
    '''Complete hybrid analysis: exercise detection + form coaching'''
    try:
        features = extract_features(coords)
        if features is None:
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
        
        if exercise in form_coaches and exercise_confidence > 0.7:
            form_model = form_coaches[exercise]
            form_encoder = form_encoders[exercise]
            
            # Use same features for form analysis (could be different in production)
            form_pred = form_model.predict(features_2d)[0]
            form_probs = form_model.predict_proba(features_2d)[0]
            form_quality = form_encoder.inverse_transform([form_pred])[0]
            form_confidence = max(form_probs)
        
        return exercise, exercise_confidence, form_quality, form_confidence
        
    except Exception as e:
        print(f'Analysis error: {e}')
        return 'unknown', 0.0, None, 0.0

def main():
    cap = cv2.VideoCapture(0)
    detector = BasePoseDetector(min_detection_confidence=0.7)
    
    print('🎥 HYBRID SYSTEM ACTIVE!')
    print('Phase 1: Exercise Detection')
    print('Phase 2: Form Coaching')
    print('Press Q to quit')
    
    prev_exercise = None
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Process every 5th frame
        if frame_count % 1 == 0:
            landmarks = detector.process(frame)
            
            if landmarks:
                coords = [(lm.x, lm.y) for lm in landmarks.landmark]
                exercise, ex_conf, form_quality, form_conf = analyze_hybrid(coords)
                
                # Alert on exercise change
                if exercise != prev_exercise and ex_conf > 0.7 and exercise != 'unknown':
                    print(f'🏋️ {exercise.upper()} detected!')
                    prev_exercise = exercise
                
                # Display Phase 1: Exercise Detection
                ex_color = (0, 255, 0) if ex_conf > 0.8 else (0, 255, 255)
                cv2.putText(frame, f'Exercise: {exercise.upper()}', (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, ex_color, 3)
                cv2.putText(frame, f'Confidence: {ex_conf:.2f}', (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, ex_color, 2)
                
                # Display Phase 2: Form Analysis
                if form_quality and form_conf > 0.5:
                    form_colors = {
                        'excellent': (0, 255, 0),    # Green
                        'good': (0, 255, 255),       # Yellow  
                        'poor': (0, 0, 255)         # Red
                    }
                    form_color = form_colors.get(form_quality, (255, 255, 255))
                    
                    cv2.putText(frame, f'Form: {form_quality.upper()}', (20, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.1, form_color, 3)
                    cv2.putText(frame, f'Quality: {form_conf:.2f}', (20, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, form_color, 2)
                    
                    # Coaching feedback
                    if form_quality == 'excellent':
                        feedback = 'Perfect form! 🏆'
                    elif form_quality == 'good':  
                        feedback = 'Good! Minor improvements needed'
                    else:
                        feedback = 'Focus on form - check technique'
                    
                    cv2.putText(frame, feedback, (20, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)
                else:
                    cv2.putText(frame, 'Analyzing form...', (20, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(frame, 'NO POSE DETECTED', (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        
        cv2.imshow('🏋️ Hybrid Exercise Coach', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print('✅ Hybrid coaching session complete!')

main()
