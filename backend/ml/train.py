# backend/ml/safe_hybrid_train.py

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import argparse
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_exercise_detector():
    """Train Phase 1: Exercise Detection (safe version)"""
    print('Training exercise detector...')
    
    dataset_files = list(Path('ml/data/processed').glob('exercise_detection_[0-9]*.csv'))
    if not dataset_files:
        print('❌ No exercise detection data found!')
        return
    
    latest_file = max(dataset_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    X = np.nan_to_num(X, 0)
    
    # Create SYNCHRONIZED encoder and model
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f'Exercise detector accuracy: {accuracy:.4f}')
    
    # Test sync
    test_pred = model.predict(X_test[:3])
    test_labels = label_encoder.inverse_transform(test_pred)
    print(f'Test: {test_pred} -> {test_labels}')
    
    # Save
    model_dir = Path('app/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / 'exercise_detector.pkl')
    joblib.dump(label_encoder, model_dir / 'exercise_detector_encoder.pkl')
    
    print('✅ Exercise detector saved!')

def train_form_coach(exercise_type):
    """Train Phase 2: Form Coach (safe version)"""
    print(f'Training {exercise_type} form coach...')
    
    dataset_files = list(Path('ml/data/processed').glob(f'{exercise_type}_form_analysis_[0-9]*.csv'))
    if not dataset_files:
        print(f'❌ No {exercise_type} form data found!')
        return
    
    latest_file = max(dataset_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    print(f'Classes: {sorted(df["label"].unique())}')
    print(f'Distribution: {df["label"].value_counts().to_dict()}')
    
    X = df.drop('label', axis=1).values
    y = df['label'].values
    X = np.nan_to_num(X, 0)
    
    # Create SYNCHRONIZED encoder and model
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    
    model = RandomForestClassifier(
        n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f'{exercise_type} form coach accuracy: {accuracy:.4f}')
    
    # Test sync
    test_pred = model.predict(X_test[:3])
    test_labels = label_encoder.inverse_transform(test_pred)
    print(f'Test: {test_pred} -> {test_labels}')
    
    # Save
    model_dir = Path('app/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f'{exercise_type}_form_coach.pkl')
    joblib.dump(label_encoder, model_dir / f'{exercise_type}_form_coach_encoder.pkl')
    
    print(f'✅ {exercise_type} form coach saved!')

def main():
    parser = argparse.ArgumentParser(description='Safe hybrid training')
    parser.add_argument('--phase', choices=['exercise-detection', 'form-analysis'], required=True)
    parser.add_argument('--exercise', type=str, help='Exercise for form analysis')
    
    args = parser.parse_args()
    
    if args.phase == 'exercise-detection':
        train_exercise_detector()
    elif args.phase == 'form-analysis':
        if not args.exercise:
            print('❌ --exercise required for form-analysis')
            return
        train_form_coach(args.exercise)

if __name__ == "__main__":
    main()