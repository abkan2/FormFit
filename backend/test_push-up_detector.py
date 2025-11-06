import asyncio
import cv2
import mediapipe as mp
import time
import sys

from app.services.detectors.calibration import BodyCalibrator
from app.services.detectors.pushup_detector import PushupDetector
from app.services.firebase_service import firebase_service

def draw_landmarks(frame, landmarks, connections):
    """Draw pose landmarks on frame"""
    height, width = frame.shape[:2]
    
    # Draw connections
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            
            if start_point and end_point:
                start_x = int(start_point[0] * width)
                start_y = int(start_point[1] * height)
                end_x = int(end_point[0] * width)
                end_y = int(end_point[1] * height)
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Draw landmarks
    for i, landmark in enumerate(landmarks):
        if landmark:
            x = int(landmark[0] * width)
            y = int(landmark[1] * height)
            
            # Different colors for key points
            if i in [0]:  # nose
                color = (0, 0, 255)
            elif i in [11, 12]:  # shoulders
                color = (255, 0, 0)
            elif i in [13, 14]:  # elbows
                color = (0, 255, 255)
            elif i in [23, 24]:  # hips
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)
            
            cv2.circle(frame, (x, y), 5, color, -1)

def display_pushup_ui(frame, detector_result):
    """Display push-up detector UI overlay"""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    # Rep counter - Big and prominent
    rep_text = f"REPS: {detector_result['rep_count']}"
    cv2.putText(overlay, rep_text, (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    
    # Phase indicator with color coding
    phase = detector_result['phase']
    phase_colors = {
        'down': (0, 165, 255),      # Orange
        'up': (0, 255, 0),          # Green
        'transition': (255, 255, 0), # Yellow
        'down_bad_form': (0, 0, 255), # Red
        'up_bad_form': (0, 0, 255),   # Red
        'neutral': (200, 200, 200),   # Gray
        'error': (0, 0, 255)          # Red
    }
    
    phase_color = phase_colors.get(phase, (255, 255, 255))
    phase_display = phase.upper().replace('_', ' ')
    
    cv2.putText(overlay, f"Phase: {phase_display}", (20, 140), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)
    
    # Calibration status
    calibration_text = "✅ CALIBRATED" if detector_result['calibrated'] else "⚠️ NOT CALIBRATED"
    calibration_color = (0, 255, 0) if detector_result['calibrated'] else (0, 165, 255)
    cv2.putText(overlay, calibration_text, (20, 180), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, calibration_color, 2)
    
    # Form percentage
    form_pct = detector_result['form_percentage']
    form_color = (0, 255, 0) if form_pct >= 80 else (0, 165, 255) if form_pct >= 60 else (0, 0, 255)
    cv2.putText(overlay, f"Form: {form_pct:.0f}%", (20, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)
    
    # Real-time feedback
    feedback = detector_result.get('feedback', '')
    if feedback:
        # Split feedback into multiple lines if too long
        words = feedback.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            test_line = ' '.join(current_line)
            text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            if text_size[0] > width - 40:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw feedback box
        feedback_y = height - 150
        box_height = len(lines) * 35 + 20
        cv2.rectangle(overlay, (10, feedback_y - 10), 
                     (width - 10, feedback_y + box_height), (0, 0, 0), -1)
        
        # Draw feedback text
        for i, line in enumerate(lines):
            cv2.putText(overlay, line, (20, feedback_y + 30 + i * 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Form issues
    if detector_result.get('form_issues'):
        issues_text = "Issues: " + ", ".join(detector_result['form_issues'][:2])
        cv2.putText(overlay, issues_text, (20, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Phase data (debug info)
    if 'phase_data' in detector_result:
        phase_data = detector_result['phase_data']
        if 'elbow_angle' in phase_data:
            angle_text = f"Elbow: {phase_data['elbow_angle']:.1f}°"
            cv2.putText(overlay, angle_text, (width - 200, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Controls
    cv2.putText(overlay, "Controls: 'q' quit | 'r' reset | 's' stats", 
               (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    return frame

def test_camera(camera_idx):
    """Test if a camera index works"""
    test_cap = cv2.VideoCapture(camera_idx)
    
    if not test_cap.isOpened():
        return False
    
    ret, test_frame = test_cap.read()
    success = ret and test_frame is not None
    
    if success:
        width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"    📐 Camera {camera_idx}: {width}x{height}")
    
    test_cap.release()
    return success

async def test_pushup_detector():
    """Test push-up detector with real camera and calibration"""
    print("🏋️ Starting Push-Up Detector Test")
    print("=" * 60)
    
    # Get user UID
    user_uid = input("Enter user UID (or press Enter for test mode): ").strip()
    
    # Initialize calibrator
    if user_uid and firebase_service.db:
        calibrator = BodyCalibrator(user_id=user_uid, firebase_client=firebase_service.db)
        print("🔍 Loading calibration data...")
        
        if calibrator.load_user_calibration():
            print("✅ Calibration loaded successfully!")
            print(f"📏 Measurements: {list(calibrator.baseline_measurements.keys())}")
        else:
            print("⚠️ No calibration found - detector will use generic thresholds")
            print("💡 Run calibration first for personalized detection!")
    else:
        print("⚠️ Running in test mode without calibration")
        calibrator = None
    
    # Initialize MediaPipe
    print("🤖 Initializing MediaPipe...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize push-up detector
    pushup_detector = PushupDetector(calibrator=calibrator)
    
    # Find camera
    print("📷 Starting camera...")
    cap = None
    
    camera_priority = [
        ([1, 2, 3, 4], "🤳 Looking for front-facing camera..."),
        ([0], "⚠️ Trying back camera..."),
    ]

    for camera_indices, message in camera_priority:
        if cap is not None:
            break
            
        print(message)
        
        for camera_idx in camera_indices:
            print(f"🔍 Trying camera index {camera_idx}...")
            
            if test_camera(camera_idx):
                cap = cv2.VideoCapture(camera_idx)
                if cap.isOpened():
                    print(f"✅ Camera {camera_idx} selected!")
                    break
        
        if cap is not None:
            break

    if cap is None:
        print("❌ No available camera found")
        return

    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✅ Camera initialized")
    print("\n🎬 PUSH-UP DETECTION STARTED")
    print("Position yourself in plank position")
    print("Make sure your full body is visible")
    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset counter")
    print("  's' - Show detailed stats")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Failed to read frame - retrying...")
                time.sleep(0.1)
                continue
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract landmarks
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y))
                
                # Update push-up detector
                detector_result = pushup_detector.update(landmarks)
                
                # Add feedback to result
                detector_result['feedback'] = pushup_detector.get_feedback()
                
                # Draw landmarks
                draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Display UI
                frame = display_pushup_ui(frame, detector_result)
                
                # Print rep completion
                if detector_result['rep_completed']:
                    print(f"\n🎉 Rep #{detector_result['rep_count']} completed!")
                    if detector_result['form_issues']:
                        print(f"   ⚠️ Issues: {', '.join(detector_result['form_issues'])}")
                    else:
                        print("   ✅ Perfect form!")
            
            else:
                # No pose detected
                cv2.putText(frame, "NO POSE DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Position yourself in plank", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Make sure full body is visible", (50, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('AI Fitness Trainer - Push-Up Detector', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Push-up detection stopped")
                break
            elif key == ord('r'):
                print("\n🔄 Resetting counter...")
                pushup_detector.reset()
            elif key == ord('s'):
                print("\n📊 Detailed Statistics:")
                stats = pushup_detector.get_stats()
                for key, value in stats.items():
                    print(f"   {key}: {value}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        pose.close()
        
        # Final stats
        print("\n" + "=" * 60)
        print("📊 FINAL STATISTICS")
        print("=" * 60)
        stats = pushup_detector.get_stats()
        print(f"Total Reps: {stats['total_reps']}")
        print(f"Good Form Reps: {stats['good_form_reps']}")
        print(f"Form Percentage: {stats['form_percentage']:.1f}%")
        print(f"Calibrated: {stats['calibrated']}")
        
        if stats.get('calibration_data'):
            print("\n📏 Calibration Data Used:")
            for key, value in stats['calibration_data'].items():
                if value is not None:
                    print(f"   {key}: {value:.4f}")
        
        print("\n🎉 Test completed!")

async def main():
    """Main test runner"""
    print("🏋️‍♂️ AI FITNESS TRAINER - PUSH-UP DETECTOR TEST")
    print("=" * 60)
    
    try:
        await test_pushup_detector()
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("📋 Pre-Test Checklist:")
    print("✅ Camera connected and working")
    print("✅ Good lighting for pose detection")
    print("✅ Enough space to do push-ups")
    print("✅ Position yourself in plank to start")
    print("✅ Make sure full body visible in frame")
    print("\nPress Enter when ready...")
    input()
    
    asyncio.run(main())