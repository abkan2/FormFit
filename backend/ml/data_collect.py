# backend/ml/data_collect.py

import argparse
from pathlib import Path
import cv2
import json
import time
import numpy as np
from datetime import datetime
import os
from app.services.mediapipe.base_detector import BasePoseDetector

class DataCollector:
        def __init__(self, exercise_name, target_samples=500, confidence_threshold=0.7):
            self.exercise_name = exercise_name
            self.target_samples = target_samples
            self.confidence_threshold = confidence_threshold
            self.detector = BasePoseDetector(min_detection_confidence=confidence_threshold)
            self.collected_data = []
            
        def run(self):
            """Simple data collection"""
            cap = cv2.VideoCapture(0)
            print(f"Collecting {self.target_samples} samples for {self.exercise_name}")
            print("Press 'q' to quit early")
            
            while len(self.collected_data) < self.target_samples:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                landmarks = self.detector.process(frame)
                
                if landmarks:
                    coords = [(lm.x, lm.y) for lm in landmarks.landmark]
                    self.collected_data.append({
                        'coordinates': coords,
                        'timestamp': time.time()
                    })
                    
                    # Draw landmarks and progress
                    frame = self.detector.draw_landmarks(frame, landmarks)
                    progress = len(self.collected_data) / self.target_samples
                    cv2.putText(frame, f"{self.exercise_name}: {len(self.collected_data)}/{self.target_samples}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                cv2.imshow('Data Collection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            
            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml/data/raw/{self.exercise_name}_{timestamp}.json"
            
            os.makedirs("ml/data/raw", exist_ok=True)
            with open(filename, 'w') as f:
                json.dump({
                    'metadata': {
                        'exercise': self.exercise_name,
                        'samples': len(self.collected_data),
                        'timestamp': timestamp
                    },
                    'samples': self.collected_data
                }, f)
            
            print(f"Saved {len(self.collected_data)} samples to {filename}")

class HybridDataCollector:
    """Specialized data collection for hybrid approach"""
    
    def __init__(self):
        self.data_types = {
            'exercise_detection': {
                'description': 'General exercise identification',
                'exercises': ['squat', 'plank', 'pushup', 'deadlift', 'rest', 'transition'],
                'samples_per_exercise': 500,
                'focus': 'Exercise variety and clear distinctions'
            },
            'form_analysis': {
                'description': 'Form quality for specific exercises',
                'exercises': ['squat_excellent', 'squat_good', 'squat_poor', 
                            'plank_excellent', 'plank_good', 'plank_poor', 
                            "pushup_excellent", "pushup_good", "pushup_poor"],
                'samples_per_exercise': 300,
                'focus': 'Form quality variations within each exercise'
            }
        }
    
    def collect_exercise_detection_data(self):
        """Phase 1: Collect data for general exercise detection"""
        print("=== PHASE 1: EXERCISE DETECTION DATA ===")
        print("Goal: Train model to distinguish between different exercises")
        print("\nCollecting data for general exercise identification...")
        
        exercises = self.data_types['exercise_detection']['exercises']
        samples = self.data_types['exercise_detection']['samples_per_exercise']
        
        for exercise in exercises:
            print(f"\n--- Collecting {exercise.upper()} data ---")
            if exercise == 'rest':
                print("Instructions: Stand/sit normally, no specific exercise")
                samples = 300  # Less rest data needed
            elif exercise == 'transition':
                print("Instructions: Move between different exercise positions")
                samples = 200  # Transition data
            else:
                print(f"Instructions: Perform {exercise} with normal form variation")
            
            collector = DataCollector(
                exercise_name=exercise,
                target_samples=samples,
                confidence_threshold=0.7
            )
            collector.run()
    
    def collect_form_analysis_data(self):
        """Phase 2: Collect data for form quality analysis"""
        print("\n=== PHASE 2: FORM ANALYSIS DATA ===")
        print("Goal: Train models to assess form quality for each exercise")
        print("\nCollecting data for form quality assessment...")
        
        form_exercises = self.data_types['form_analysis']['exercises']
        samples = self.data_types['form_analysis']['samples_per_exercise']
        
        for exercise in form_exercises:
            print(f"\n--- Collecting {exercise.upper()} data ---")
            
            # Provide specific instructions for each form quality
            if 'excellent' in exercise:
                print("Instructions: Perfect form - full range of motion, proper alignment")
            elif 'good' in exercise:
                print("Instructions: Good form with minor imperfections")
            elif 'poor' in exercise:
                print("Instructions: Common form mistakes (shallow depth, poor alignment, etc.)")
            
            collector = DataCollector(
                exercise_name=exercise,
                target_samples=samples,
                confidence_threshold=0.6  # Lower threshold for form variations
            )
            collector.run()
    
    def collect_progressive_data(self, exercise_base: str):
        """Collect progressive form data for a specific exercise"""
        print(f"\n=== PROGRESSIVE COLLECTION: {exercise_base.upper()} ===")
        
        form_levels = ['excellent', 'good', 'needs_improvement', 'poor']
        instructions = {
            'excellent': "Perfect form - demonstrate ideal technique",
            'good': "Good form with minor room for improvement", 
            'needs_improvement': "Noticeable form issues but not terrible",
            'poor': "Common beginner mistakes and form breakdowns"
        }
        
        for level in form_levels:
            exercise_name = f"{exercise_base}_{level}"
            print(f"\n--- {exercise_name.upper()} ---")
            print(f"Instructions: {instructions[level]}")
            
            collector = DataCollector(
                exercise_name=exercise_name,
                target_samples=250,
                confidence_threshold=0.6
            )
            collector.run()

def print_data_collection_plan():
    """Print the complete data collection strategy"""
    print("="*60)
    print("HYBRID APPROACH DATA COLLECTION PLAN")
    print("="*60)
    
    print("\n🎯 PHASE 1: EXERCISE DETECTION (General Model)")
    print("├── squat (500 samples) - Various squat forms")
    print("├── plank (500 samples) - Various plank forms") 
    print("├── pushup (500 samples) - Various pushup forms")
    print("├── deadlift (500 samples) - Various deadlift forms")
    print("├── rest (300 samples) - Standing/sitting normally")
    print("└── transition (200 samples) - Moving between exercises")
    print("    Total: 2,500 samples for exercise identification")
    
    print("\n🔍 PHASE 2: FORM ANALYSIS (Specialized Models)")
    print("├── Squat Form Coach:")
    print("│   ├── squat_excellent (300 samples) - Perfect depth, alignment")
    print("│   ├── squat_good (300 samples) - Minor form issues") 
    print("│   └── squat_poor (300 samples) - Shallow, poor alignment")
    print("├── Plank Form Coach:")
    print("│   ├── plank_excellent (300 samples) - Perfect body line")
    print("│   ├── plank_good (300 samples) - Minor sagging/misalignment")
    print("│   └── plank_poor (300 samples) - Major form breakdown")
    print("└── Future exercises: pushup, deadlift form coaches...")
    print("    Total: 1,800+ samples for form quality assessment")
    
    print("\n⏱️ ESTIMATED TIME INVESTMENT:")
    print("├── Phase 1 (Exercise Detection): ~4-5 hours")
    print("├── Phase 2 (Form Analysis): ~3-4 hours")  
    print("└── Total Data Collection: ~7-9 hours")
    
    print("\n📈 EXPECTED RESULTS:")
    print("├── Exercise Detection: 90-95% accuracy")
    print("├── Form Analysis: 80-90% accuracy")
    print("└── Combined System: Premium coaching experience")
    
    print("\n QUICK START COMMANDS:")
    print("# Collect exercise detection data")
    print("python hybrid_data_collect.py --phase exercise-detection")
    print("\n# Collect form analysis data")  
    print("python hybrid_data_collect.py --phase form-analysis")
    print("\n# Collect progressive data for specific exercise")
    print("python hybrid_data_collect.py --progressive squat")

def main():
    parser = argparse.ArgumentParser(description='Hybrid approach data collection')
    parser.add_argument('--phase', choices=['exercise-detection', 'form-analysis', 'plan'],
                       help='Data collection phase')
    parser.add_argument('--progressive', type=str, 
                       help='Collect progressive form data for specific exercise (e.g., squat)')
    parser.add_argument('--exercise', type=str,
                       help='Collect data for single exercise')
    parser.add_argument('--samples', type=int, default=500,
                       help='Number of samples to collect')
    
    args = parser.parse_args()
    
    collector = HybridDataCollector()
    
    if args.phase == 'plan':
        print_data_collection_plan()
    elif args.phase == 'exercise-detection':
        collector.collect_exercise_detection_data()
    elif args.phase == 'form-analysis':
        collector.collect_form_analysis_data()
    elif args.progressive:
        collector.collect_progressive_data(args.progressive)
    elif args.exercise:
        print(f"Collecting data for {args.exercise}")
        single_collector = DataCollector(
            exercise_name=args.exercise,
            target_samples=args.samples,
            confidence_threshold=0.7
        )
        single_collector.run()
    else:
        print_data_collection_plan()
        print("\nUse --phase to start data collection")

if __name__ == "__main__":
    main()