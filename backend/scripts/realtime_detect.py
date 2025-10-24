# Simplified realtime_detect.py using rule-based detection

import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve()))

import cv2
import time
from app.services.mediapipe.base_detector import BasePoseDetector
from app.services.detectors.pushup_detector import PushupDetector

def main():
    print('🎥 Starting Rule-Based Pushup Detection...')
    
    # Initialize detectors
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
        
    pose_detector = BasePoseDetector(min_detection_confidence=0.7)
    pushup_detector = PushupDetector()
    
    print('✅ Ready! Do some pushups!')
    print('Press Q to quit, R to reset counter')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Detect pose
        landmarks = pose_detector.process(frame)
        
        if landmarks:
            coords = pose_detector.extract_ml_coordinates(landmarks)
            quality = pose_detector.get_landmark_quality(landmarks)
            
            if quality > 0.6:
                # Update pushup detection
                result = pushup_detector.update(coords)
                
                # Display results
                cv2.putText(frame, f"Phase: {result['phase'].upper()}", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Reps: {result['rep_count']}", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"Quality: {quality:.2f}", 
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw landmarks
                frame = pose_detector.draw_minimal_landmarks(frame, landmarks)
                
                # Rep completion feedback
                if result['rep_completed']:
                    cv2.putText(frame, "REP COMPLETED!", (frame.shape[1]//2-100, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(frame, f"Low Quality: {quality:.2f}", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Pose Detected", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.imshow('Rule-Based Pushup Counter', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            pushup_detector.reset()
            print("Counter reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()

if __name__ == "__main__":
    main()