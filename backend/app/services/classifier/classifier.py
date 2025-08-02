# app/services/classifier/classifier.py
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class HybridFitnessClassifier:
    """
    Modern hybrid classifier for exercise detection and form analysis
    Phase 1: Exercise Detection (squat, plank, pushup, etc.)
    Phase 2: Form Coaching (excellent, good, poor)
    """
    
    def __init__(self):
        self.model_dir = Path(__file__).parents[2] / "models"
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load exercise detector and form coaches"""
        try:
            print('🔧 Loading hybrid classifier system...')
            
            # Load Phase 1: Exercise Detection
            self.exercise_model = joblib.load(self.model_dir / 'exercise_detector.pkl')
            self.exercise_encoder = joblib.load(self.model_dir / 'exercise_detector_encoder.pkl')
            
            # Load Phase 2: Form Coaches
            self.form_coaches = {}
            self.form_encoders = {}
            
            for exercise in ['squat', 'plank', 'pushup']:
                try:
                    self.form_coaches[exercise] = joblib.load(self.model_dir / f'{exercise}_form_coach.pkl')
                    self.form_encoders[exercise] = joblib.load(self.model_dir / f'{exercise}_form_coach_encoder.pkl')
                    print(f'✅ {exercise} form coach loaded')
                except FileNotFoundError:
                    print(f'❌ {exercise} form coach not found')
            
            # Load scaler
            scaler_paths = [
                Path('ml/data/processed'),
                self.model_dir
            ]
            
            scaler_file = None
            for path in scaler_paths:
                scaler_files = list(path.glob('*scaler*.pkl'))
                if scaler_files:
                    scaler_file = max(scaler_files, key=lambda x: x.stat().st_mtime)
                    break
            
            if scaler_file:
                self.scaler = joblib.load(scaler_file)
                print(f'✅ Scaler loaded: {scaler_file.name}')
            else:
                raise FileNotFoundError("No scaler found in expected locations")
            
            self.models_loaded = True
            print(f'✅ Hybrid classifier ready! Exercises: {list(self.exercise_encoder.classes_)}')
            
        except Exception as e:
            print(f'❌ Failed to load hybrid models: {e}')
            self.models_loaded = False
            raise RuntimeError(f"Could not initialize hybrid classifier: {e}")
    
    def extract_features(self, landmarks) -> Optional[np.ndarray]:
        """Extract normalized features from pose landmarks"""
        if isinstance(landmarks, list):
            # Convert from [(x1,y1), (x2,y2), ...] to numpy array
            coords = np.array(landmarks)
        else:
            coords = landmarks
        
        # Handle different input formats
        if coords.shape == (66,):  # Flattened [x1,y1,x2,y2,...]
            coords = coords.reshape(33, 2)
        elif coords.shape != (33, 2):
            print(f"⚠️ Invalid landmark shape: {coords.shape}, expected (33, 2)")
            return None
        
        features = []
        
        # Normalize pose relative to body size
        shoulder_width = np.linalg.norm(coords[11] - coords[12])
        if shoulder_width <= 0:
            return None
            
        hip_center = (coords[23] + coords[24]) / 2
        normalized_coords = (coords - hip_center) / shoulder_width
        
        # Key landmarks for pose representation
        key_landmarks = {
            'nose': 0, 'left_eye': 2, 'right_eye': 5,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        # Add normalized coordinates
        for idx in key_landmarks.values():
            features.extend(normalized_coords[idx])
        
        # Add engineered features for better classification
        try:
            shoulder_center = (coords[11] + coords[12]) / 2
            torso_vector = shoulder_center - hip_center
            torso_angle = np.arctan2(torso_vector[1], torso_vector[0]) * 180 / np.pi
            features.append(torso_angle)
            
            head_to_hip = np.linalg.norm(coords[0] - hip_center)
            compactness = head_to_hip / shoulder_width
            features.append(compactness)
            
            def calc_angle(p1, p2, p3):
                v1, v2 = p1 - p2, p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            # Joint angles for pose characterization
            left_arm_angle = calc_angle(coords[11], coords[13], coords[15])
            right_arm_angle = calc_angle(coords[12], coords[14], coords[16])
            left_leg_angle = calc_angle(coords[23], coords[25], coords[27])
            right_leg_angle = calc_angle(coords[24], coords[26], coords[28])
            
            features.extend([left_arm_angle, right_arm_angle, left_leg_angle, right_leg_angle])
            
        except (ZeroDivisionError, ValueError):
            # Handle edge cases in angle calculation
            features.extend([0.0] * 6)
        
        # Ensure consistent feature vector length
        while len(features) < 36:
            features.append(0.0)
            
        return np.array(features[:36])
    
    def predict_exercise(self, landmarks) -> Tuple[str, float]:
        """
        Phase 1: Predict exercise type
        Returns: (exercise_name, confidence)
        """
        if not self.models_loaded:
            return 'unknown', 0.0
        
        try:
            features = self.extract_features(landmarks)
            if features is None:
                return 'unknown', 0.0
            
            features_2d = features.reshape(1, -1)
            features_2d = self.scaler.transform(features_2d)
            
            exercise_pred = self.exercise_model.predict(features_2d)[0]
            exercise_probs = self.exercise_model.predict_proba(features_2d)[0]
            exercise = self.exercise_encoder.inverse_transform([exercise_pred])[0]
            confidence = max(exercise_probs)
            
            return exercise, confidence
            
        except Exception as e:
            print(f'Exercise prediction error: {e}')
            return 'unknown', 0.0
    
    def predict_form(self, landmarks, exercise: str) -> Tuple[Optional[str], float]:
        """
        Phase 2: Predict form quality for specific exercise
        Returns: (form_quality, confidence)
        """
        if not self.models_loaded or exercise not in self.form_coaches:
            return None, 0.0
        
        try:
            features = self.extract_features(landmarks)
            if features is None:
                return None, 0.0
            
            features_2d = features.reshape(1, -1)
            features_2d = self.scaler.transform(features_2d)
            
            form_model = self.form_coaches[exercise]
            form_encoder = self.form_encoders[exercise]
            
            form_pred = form_model.predict(features_2d)[0]
            form_probs = form_model.predict_proba(features_2d)[0]
            form_quality = form_encoder.inverse_transform([form_pred])[0]
            confidence = max(form_probs)
            
            return form_quality, confidence
            
        except Exception as e:
            print(f'Form prediction error for {exercise}: {e}')
            return None, 0.0
    
    def analyze(self, landmarks) -> Tuple[str, float, Optional[str], float]:
        """
        Complete hybrid analysis: exercise detection + form coaching
        Returns: (exercise, exercise_confidence, form_quality, form_confidence)
        """
        # Phase 1: Exercise Detection
        exercise, exercise_confidence = self.predict_exercise(landmarks)
        
        # Phase 2: Form Analysis (only if exercise detection is confident)
        form_quality, form_confidence = None, 0.0
        if exercise_confidence > 0.7:
            form_quality, form_confidence = self.predict_form(landmarks, exercise)
        
        return exercise, exercise_confidence, form_quality, form_confidence
    
    @property
    def supported_exercises(self) -> list:
        """Get list of supported exercises"""
        if hasattr(self, 'exercise_encoder'):
            return list(self.exercise_encoder.classes_)
        return []
    
    @property
    def form_coached_exercises(self) -> list:
        """Get list of exercises with form coaching"""
        return list(self.form_coaches.keys()) if hasattr(self, 'form_coaches') else []

# Global classifier instance
_classifier = None

def get_classifier() -> HybridFitnessClassifier:
    """Get the global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = HybridFitnessClassifier()
    return _classifier

def analyze_pose(landmarks):
    """
    Main API function for pose analysis
    Returns: (exercise, exercise_confidence, form_quality, form_confidence)
    """
    return get_classifier().analyze(landmarks)

def predict_exercise(landmarks):
    """Predict exercise type only"""
    return get_classifier().predict_exercise(landmarks)

def predict_form(landmarks, exercise: str):
    """Predict form quality for specific exercise"""
    return get_classifier().predict_form(landmarks, exercise)

# Initialize on import
clf = get_classifier()
print(f"🏋️ Hybrid classifier ready: {clf.supported_exercises}")
print(f"📊 Form coaching for: {clf.form_coached_exercises}")

if __name__ == "__main__":
    # Test the classifier
    print(f"Supported exercises: {clf.supported_exercises}")
    print(f"Form coaches: {clf.form_coached_exercises}")
    print(f"Models loaded: {clf.models_loaded}")