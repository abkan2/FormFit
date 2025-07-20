
# Let's create a new preprocessing script that uses 36 features for everything
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def normalize_pose(coordinates):
    '''Normalize pose to be body-size invariant'''
    try:
        coords_array = np.array(coordinates)
        shoulder_width = np.linalg.norm(coords_array[11] - coords_array[12])
        if shoulder_width > 0:
            hip_center = (coords_array[23] + coords_array[24]) / 2
            normalized = (coords_array - hip_center) / shoulder_width
            return normalized
    except:
        pass
    return np.array(coordinates)

def calculate_angle(p1, p2, p3):
    '''Calculate angle between three points'''
    try:
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    except:
        return 90.0

def extract_36_features(coordinates):
    '''Extract exactly 36 features (same for exercise detection AND form analysis)'''
    coords_array = np.array(coordinates)
    if coords_array.shape != (33, 2):
        return []
    
    features = []
    
    # 1. Normalized coordinates (body-size invariant)
    normalized_coords = normalize_pose(coordinates)
    
    # Use only key landmarks (reduces dimensionality but keeps important info)
    key_landmarks = {
        'nose': 0, 'left_eye': 2, 'right_eye': 5,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28
    }
    
    for landmark_name, idx in key_landmarks.items():
        if idx < len(normalized_coords):
            features.extend(normalized_coords[idx])  # x, y
    
    # 2. Key body ratios and angles for exercise identification
    try:
        # Torso orientation (standing vs lying)
        shoulder_center = (coords_array[11] + coords_array[12]) / 2
        hip_center = (coords_array[23] + coords_array[24]) / 2
        torso_vector = shoulder_center - hip_center
        torso_angle = np.arctan2(torso_vector[1], torso_vector[0]) * 180 / np.pi
        features.append(torso_angle)
        
        # Body compactness (squat vs standing)
        head_to_hip = np.linalg.norm(coords_array[0] - hip_center)
        shoulder_width = np.linalg.norm(coords_array[11] - coords_array[12])
        if shoulder_width > 0:
            compactness = head_to_hip / shoulder_width
            features.append(compactness)
        else:
            features.append(0.0)
        
        # Limb positions relative to torso
        left_arm_angle = calculate_angle(coords_array[11], coords_array[13], coords_array[15])
        right_arm_angle = calculate_angle(coords_array[12], coords_array[14], coords_array[16])
        features.extend([left_arm_angle, right_arm_angle])
        
        # Legs relative to hips
        left_leg_angle = calculate_angle(coords_array[23], coords_array[25], coords_array[27])
        right_leg_angle = calculate_angle(coords_array[24], coords_array[26], coords_array[28])
        features.extend([left_leg_angle, right_leg_angle])
        
    except Exception as e:
        # Pad with zeros if calculation fails
        features.extend([0.0] * 6)
    
    # Ensure exactly 36 features
    while len(features) < 36:
        features.append(0.0)
    features = features[:36]
    
    return features

def load_json_data(json_file):
    '''Load and validate JSON data'''
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            samples = [{'coordinates': coords} for coords in data]
            metadata = {'format': 'legacy', 'file': json_file.name}
        else:
            samples = data.get('samples', [])
            metadata = data.get('metadata', {})
            metadata['file'] = json_file.name
        
        print(f'Loaded {len(samples)} samples from {json_file.name}')
        return {'samples': samples, 'metadata': metadata}
        
    except Exception as e:
        print(f'Error loading {json_file}: {e}')
        return {'samples': [], 'metadata': {}}

def process_form_data_36_features(exercise_type):
    '''Process form analysis data with 36 features'''
    print(f'Processing {exercise_type} form data with 36 features...')
    
    form_categories = [f'{exercise_type}_excellent', f'{exercise_type}_good', f'{exercise_type}_poor']
    raw_dir = Path('ml/data/raw')
    
    all_features = []
    all_labels = []
    
    for form_label in form_categories:
        pattern = f'*{form_label}*.json'
        matching_files = list(raw_dir.glob(pattern))
        
        for json_file in matching_files:
            if 'coords_only' in json_file.name:
                continue
            
            data = load_json_data(json_file)
            samples = data['samples']
            
            for sample in samples:
                try:
                    if 'coordinates' in sample:
                        coords = sample['coordinates']
                        if len(coords) == 33:
                            features = extract_36_features(coords)
                            if len(features) == 36:
                                all_features.append(features)
                                
                                # Map to quality label
                                if 'excellent' in form_label:
                                    quality = 'excellent'
                                elif 'good' in form_label:
                                    quality = 'good'
                                else:
                                    quality = 'poor'
                                all_labels.append(quality)
                except Exception as e:
                    continue
    
    if not all_features:
        print(f'No form analysis features extracted for {exercise_type}!')
        return
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(36)]
    df = pd.DataFrame(all_features, columns=feature_names)
    df['label'] = all_labels
    
    print(f'{exercise_type} form dataset: {len(df)} samples, 36 features')
    # print(f'Quality distribution: {df[\"label\"].value_counts().to_dict()}')
    
    # Save processed data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    processed_dir = Path('ml/data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and normalize
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    feature_cols = [col for col in df_shuffled.columns if col != 'label']
    scaler = StandardScaler()
    df_shuffled[feature_cols] = scaler.fit_transform(df_shuffled[feature_cols])
    
    # Save
    dataset_path = processed_dir / f'{exercise_type}_form_36feat_{timestamp}.csv'
    scaler_path = processed_dir / f'{exercise_type}_form_36feat_scaler_{timestamp}.pkl'
    
    df_shuffled.to_csv(dataset_path, index=False)
    joblib.dump(scaler, scaler_path)
    
    print(f'✅ Saved: {dataset_path}')
    print(f'✅ Saved: {scaler_path}')

# Process all form data with 36 features
for exercise in ['squat', 'plank', 'pushup']:
    process_form_data_36_features(exercise)

print('\\n✅ All form data reprocessed with 36 features!')
