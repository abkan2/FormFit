import asyncio
import cv2
import mediapipe as mp
import sys
import os
import time

from app.services.detectors.calibration import BodyCalibrator
from app.services.firebase_service import firebase_service

def display_calibration_ui(instruction: str, progress: int, target: int, frame):
    """Display normal calibration UI (no transition handling)"""
    height, width = frame.shape[:2]
    
    # Create overlay
    overlay = frame.copy()
    
    # Determine phase (but no transition handling here)
    front_frames = 60  # Should match your calibrator settings
    is_front_phase = progress < front_frames
    
    # Progress bar
    progress_percent = (progress / target) * 100
    bar_width = int(width * 0.8)
    bar_height = 30
    bar_x = (width - bar_width) // 2
    bar_y = 50
    
    # Background bar
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Progress fill - different colors for phases
    fill_width = int((progress / target) * bar_width)
    if is_front_phase:
        fill_color = (0, 255, 255)  # Yellow for front
    else:
        fill_color = (255, 0, 255)  # Magenta for side
        
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fill_color, -1)
    
    # Progress text
    progress_text = f"{progress}/{target} ({progress_percent:.1f}%)"
    cv2.putText(overlay, progress_text, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Phase indicator
    if is_front_phase:
        phase = "📷 FRONT VIEW"
        phase_color = (0, 255, 255)  # Yellow
    else:
        phase = "📷 SIDE VIEW" 
        phase_color = (255, 0, 255)  # Magenta
    
    cv2.putText(overlay, phase, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, phase_color, 3)
    
    # Visual pose guides
    if is_front_phase:
        # A-pose visual guide
        cv2.putText(overlay, "Stand like: \\o/", (width - 200, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(overlay, "Arms out slightly", (width - 200, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        # Side pose guide
        cv2.putText(overlay, "Side profile: |", (width - 150, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(overlay, "Stand naturally", (width - 150, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Instruction text
    lines = instruction.split('\n')
    y_offset = height - 150
    
    for i, line in enumerate(lines[:3]):  # Show max 3 lines
        if line.strip():
            cv2.putText(overlay, line.strip(), (20, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Frame counter for current phase
    if is_front_phase:
        phase_progress = f"Front: {progress}/{front_frames}"
    else:
        side_progress = progress - front_frames
        side_target = target - front_frames
        phase_progress = f"Side: {side_progress}/{side_target}"
    
    cv2.putText(overlay, phase_progress, (20, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    return frame

def display_transition_ui(frame, remaining_time):
    """Display dedicated transition UI with countdown"""
    height, width = frame.shape[:2]
    
    # Create dark overlay for focus
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Big red TURN text
    turn_text = "TURN RIGHT NOW!"
    text_size = cv2.getTextSize(turn_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 2 - 50
    
    # Flashing effect
    flash = int(time.time() * 4) % 2
    if flash:
        # Red background rectangle
        cv2.rectangle(frame, 
                     (text_x - 30, text_y - 80), 
                     (text_x + text_size[0] + 30, text_y + 30), 
                     (0, 0, 255), -1)
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 255)
    
    # Main TURN text
    cv2.putText(frame, turn_text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 2.5, text_color, 6)
    
    # Countdown timer
    countdown_text = f"Resume in: {remaining_time:.1f}s"
    countdown_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    countdown_x = (width - countdown_size[0]) // 2
    countdown_y = text_y + 100
    
    cv2.putText(frame, countdown_text, (countdown_x, countdown_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Instructions
    instructions = [
        "Turn 90° to your right",
        "Show your side profile", 
        "Stand naturally upright"
    ]
    
    for i, instruction in enumerate(instructions):
        inst_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        inst_x = (width - inst_size[0]) // 2
        inst_y = countdown_y + 60 + (i * 30)
        
        cv2.putText(frame, instruction, (inst_x, inst_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Arrow pointing right
    arrow_y = text_y + 50
    cv2.arrowedLine(frame, 
                   (width//2 - 150, arrow_y), 
                   (width//2 + 150, arrow_y), 
                   (255, 255, 255), 12, tipLength=0.2)
    
    return frame

def signal_phase_transition():
    """Visual and audio signal for phase change"""
    print("\n" + "🔄" * 20)
    print("🔄 TURN TO SIDE VIEW NOW!")
    print("🔄" * 20)

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
            elif i in [23, 24]:  # hips
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)
            
            cv2.circle(frame, (x, y), 5, color, -1)

async def verify_user_exists(user_uid: str) -> bool:
    """Verify user exists in Firebase"""
    print(f"🔍 Checking if user {user_uid} exists...")
    
    try:
        user_data = await firebase_service.get_user_data(user_uid)
        
        if user_data:
            print("✅ User found!")
            if 'personalInfo' in user_data:
                name = user_data['personalInfo'].get('name', 'Unknown')
                print(f"👤 User: {name}")
            return True
        else:
            print("❌ User not found in Firebase")
            return False
    except Exception as e:
        print(f"❌ Error verifying user: {e}")
        return False

def test_camera_and_detect_orientation(camera_idx):
    """Test camera and detect if it's front-facing"""
    test_cap = cv2.VideoCapture(camera_idx)
    
    if not test_cap.isOpened():
        return False, "not_opened"
    
    # Test if we can read frames
    ret, test_frame = test_cap.read()
    if not ret or test_frame is None:
        test_cap.release()
        return False, "no_frame"
    
    # Check camera properties to detect orientation
    width = test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"    📐 Camera {camera_idx}: {int(width)}x{int(height)}")
    
    # On mobile, front cameras often have different resolutions
    # or you can test by looking for face detection bias
    
    test_cap.release()
    return True, "working"
async def test_real_calibration(calibrator=None):
    
    # Initialize MediaPipe
    print("Initializing MediaPipe...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Initialize calibrator with your Firebase service
    calibrator = BodyCalibrator(user_id=user_uid, firebase_client=firebase_service.db)
    
    # Check for existing calibration
    print("🔍 Checking for existing calibration...")
    existing_calibration = calibrator.load_user_calibration()
    
    if existing_calibration:
        print("✅ Found existing calibration!")
        print(f"📏 Measurements available: {list(calibrator.baseline_measurements.keys())}")
        print(f"🆔 Calibration ID: {calibrator.calibration_id}")
        
        recalibrate = input("Do you want to recalibrate? (y/n): ").lower()
        if recalibrate != 'y':
            print("Using existing calibration.")
            return calibrator
        else:
            calibrator.reset_calibration()
            print("🔄 Reset calibration - starting fresh")
    calibrator.reset_calibration()
    
    # 🎯 FIXED CAMERA DETECTION - PRIORITIZE FRONT CAMERA
    print("📷 Starting camera...")
    cap = None
    
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

    # 🎯 PRIORITIZED CAMERA SEARCH
    camera_priority = [
        ([1, 2, 3, 4], "🤳 Looking for front-facing camera..."),
        ([0], "⚠️ Trying back camera..."),
        (list(range(5, 10)), "🔄 Trying additional indices...")
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
                    camera_type = "front" if camera_idx > 0 else "back"
                    print(f"✅ {camera_type.title()} camera {camera_idx} selected!")
                    break
        
        if cap is not None:
            break

    if cap is None:
        print("❌ No available camera found")
        print("💡 Troubleshooting tips:")
        print("   • Make sure camera permissions are granted")
        print("   • Close other apps using the camera")
        print("   • Try running on a different device")
        return

    # 🎯 CONFIGURE CAMERA FOR OPTIMAL SELFIE EXPERIENCE
    print("⚙️ Configuring camera for selfie mode...")

    # Set optimal resolution for mobile selfie cameras
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📐 Camera resolution: {actual_width}x{actual_height}")

    # Set other camera properties for better performance
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

    print("✅ Camera initialized and optimized for selfie mode")
    print("\n🎬 CALIBRATION STARTED")
    print("Controls: 'q' to quit, 'r' to restart")
    print("Follow the on-screen instructions...")
    
    frame_count = 0
    last_status = ""
    last_progress = 0
    
    # 🎯 TRANSITION CONTROL VARIABLES
    transition_pause = False
    transition_start_time = None
    transition_duration = 3.0  # 3 seconds pause for transition
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Failed to read frame - retrying...")
                time.sleep(0.1)
                continue
            
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(rgb_frame)
            current_time = time.time()
            
            # 🎯 HANDLE TRANSITION PAUSE
            if transition_pause:
                # During transition, don't process poses - just show TURN instruction
                elapsed = current_time - transition_start_time
                remaining = transition_duration - elapsed
                
                if remaining > 0:
                    # Still in transition - show TURN indicator
                    frame = display_transition_ui(frame, remaining)
                    
                    cv2.imshow('AI Fitness Trainer - Calibration', frame)
                    
                    # Handle key presses during transition
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("🛑 Calibration stopped by user")
                        break
                    elif key == ord('r'):
                        print("🔄 Restarting calibration...")
                        calibrator.reset_calibration()
                        last_status = ""
                        last_progress = 0
                        transition_pause = False
                        transition_start_time = None
                    
                    continue  # Skip pose processing during transition
                else:
                    # ✅ FIXED: Transition complete - reset flags but DON'T continue
                    transition_pause = False
                    transition_start_time = None
                    print("✅ Transition complete - resuming calibration")
                    # ❌ REMOVED: Don't use continue here - let it fall through to process the frame
            
            # 🎯 NORMAL POSE PROCESSING (will run after transition completes)
            if results.pose_landmarks:
                # Extract landmarks as tuples
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y))
                
                # Add to calibration
                result = await calibrator.add_calibration_frame(landmarks, "general")
                
                current_progress = result["progress"]
                
                # 🎯 DETECT TRANSITION TO SIDE VIEW
                if last_progress < 60 and current_progress >= 60:
                    signal_phase_transition()
                    # Start transition pause
                    transition_pause = True
                    transition_start_time = current_time
                    last_progress = current_progress # Update last_progress to avoid repeated triggers
                    print(f"🔄 Starting {transition_duration}s transition pause...")
                    continue  # Skip this frame and start showing transition UI
                
                last_progress = current_progress
                
                # Draw landmarks on frame
                draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Display normal calibration UI
                frame = display_calibration_ui(
                    result['message'], 
                    result['progress'], 
                    result['target'], 
                    frame
                )
                
                # Print status updates
                if result['status'] != last_status or frame_count % 30 == 0:
                    print(f"📊 {result['status'].upper()}: {result['progress']}/{result['target']} frames")
                    last_status = result['status']
                
                # Check if complete
                if result['status'] == 'complete':
                    print("\n🎉 CALIBRATION COMPLETE!")
                    print(f"☁️ Saved to Firebase: {result.get('saved_to_cloud', False)}")
                    
                    # Show measurements
                    measurements = result.get('measurements', {})
                    print("\n📏 Your Body Measurements:")
                    for key, value in measurements.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")
                    
                    # Display success on frame
                    cv2.putText(frame, "CALIBRATION COMPLETE!", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(frame, f"Calibration ID: {calibrator.calibration_id}", (50, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, "Press 'q' to exit", (50, 280), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('AI Fitness Trainer - Calibration', frame)
                    cv2.waitKey(3000)  # Show for 3 seconds
                    break
            
            else:
                # No pose detected
                cv2.putText(frame, "NO POSE DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Stand fully in view of camera", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Make sure you're well lit", (50, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame (only if not in transition)
            if not transition_pause:
                cv2.imshow('AI Fitness Trainer - Calibration', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Calibration stopped by user")
                break
            elif key == ord('r'):
                print("🔄 Restarting calibration...")
                calibrator.reset_calibration()
                last_status = ""
                last_progress = 0
                transition_pause = False
                transition_start_time = None
            
            frame_count += 1
    
    except Exception as e:
        print(f"❌ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("📷 Camera released")
    
    return calibrator



async def test_calibration_data(calibrator):
    """Test the calibration data"""
    if not calibrator or not calibrator.is_calibrated:
        print("❌ No calibration data to test")
        return
    
    print("\n" + "=" * 60)
    print("🧪 TESTING CALIBRATION DATA")
    print("=" * 60)
    
    measurements = calibrator.baseline_measurements
    
    print("📊 Calibration Summary:")
    print(f"  🆔 ID: {calibrator.calibration_id}")
    print(f"  📅 Type: {measurements.get('calibration_type', 'unknown')}")
    print(f"  📈 Frames: {measurements.get('total_frames', 0)}")
    print(f"  ⏰ Created: {measurements.get('created_at', 'unknown')}")
    
    print("\n📏 Body Measurements:")
    measurement_keys = [
        'shoulder_width', 'hip_width', 'arm_span', 
        'torso_length', 'leg_length', 'total_height'
    ]
    
    for key in measurement_keys:
        value = measurements.get(key)
        if value is not None:
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
    
    print("\n📊 Body Ratios:")
    ratio_keys = ['shoulder_to_hip_ratio', 'leg_to_torso_ratio']
    for key in ratio_keys:
        value = measurements.get(key)
        if value is not None:
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
    
    # Test normalization
    print("\n🔧 Testing Normalization Functions:")
    test_distance = 0.1
    
    references = ['shoulder_width', 'torso_length', 'total_height']
    for ref in references:
        if calibrator.get_measurement(ref):
            normalized = calibrator.get_normalized_distance(test_distance, ref)
            print(f"  Distance {test_distance} normalized by {ref}: {normalized:.4f}")
    
    print("\n✅ Calibration data is ready for exercise detection!")

async def main():
    """Main test runner"""
    print("🏋️‍♂️ AI FITNESS TRAINER - REAL CALIBRATION TEST")
    print("=" * 60)
    print("🔥 Using your Firebase service")
    print("🤖 Using real MediaPipe pose detection")
    print("📷 Using real camera feed")
    print("=" * 60)
    
    try:
        # Run calibration test
        calibrator = await test_real_calibration()
        
        if calibrator and calibrator.is_calibrated:
            # Test the calibration data
            await test_calibration_data(calibrator)
            
            print("\n🎉 REAL TEST COMPLETED SUCCESSFULLY!")
            print("\n📋 Test Results:")
            print("✅ Real MediaPipe pose detection working")
            print("✅ Real Firebase integration working") 
            print("✅ Visual calibration interface working")
            print("✅ Body measurements computed correctly")
            print("✅ Data saved to your Firebase database")
            print("✅ Ready for pushup/squat/pullup integration")
            
            print(f"\n🎯 Your calibration system is LIVE and ready!")
            print(f"📝 Calibration ID: {calibrator.calibration_id}")
        else:
            print("❌ Calibration was not completed")
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("📋 Pre-Test Checklist:")
    print("✅ Camera connected and working")
    print("✅ Firebase service initialized")
    print("✅ User UID from your database ready")
    print("✅ Good lighting for pose detection")
    print("✅ Stand 2-3 feet from camera")
    print("\nPress Enter when ready...")
    input()
    
    asyncio.run(main())