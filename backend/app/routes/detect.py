from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any
import cv2
import numpy as np
import mediapipe as mp
import asyncio
import json
from datetime import datetime
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCDataChannel
from av import VideoFrame
import time
from test_calibration import test_real_calibration
from app.routes.auth import require_firebase_user
from app.services.detectors.calibration import BodyCalibrator , run_calibration_session
from app.services.detectors.pushup_detector import PushupDetector
from app.services.firebase_service import firebase_service

import uuid

pcs = set()
router = APIRouter()

# Initialize MediaPipe
mp_pose = mp.solutions.pose
# Add global storage for calibration sessions
active_calibration_sessions: Dict[str, Dict[str, Any]] = {}
SESSION_TTL = 15 * 60  # 15 minutes
# ================================
# WEBRTC VIDEO TRACK
# ================================

class DetectionVideoTrack(VideoStreamTrack):
    """
    Processes frames and sends detection data via data channel
    """
    def __init__(self, track, detector_type="calibration", user_id=None, data_channel=None, session_id: Optional[str]= None):
        super().__init__()
        self.track = track
        self.detector_type = detector_type
        self.user_id = user_id
        self.data_channel = data_channel 
        self.session_id = session_id
        
        # Initialize MediaPipe
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize detector based on type
        ##Todo : handle session_id for pushups too.
        if detector_type == "pushup":
            calibrator = BodyCalibrator(
                user_id=user_id,
                firebase_client=firebase_service.db
            )
            calibrator.load_user_calibration()
            self.detector = PushupDetector(calibrator=calibrator)
            print(f"✅ Push-up detector initialized for user {user_id}")
            
        elif detector_type == "calibration":
            if session_id and session_id in active_calibration_sessions:
                session = active_calibration_sessions[session_id]
                self.detector = session["calibrator"]
                session["status"] = "active"
                print(f"✅ Using existing calibration session {session_id} for user {user_id}")
            else:
                self.detector = BodyCalibrator(
                    user_id=user_id,
                    firebase_client=firebase_service.db
                )
                print(f"✅ Calibration detector initialized for user {user_id} (new)")
        
        self.frame_count = 0
        self.last_rep_count = 0
        
        # ✅ Calibration pause tracking
        self.transition_triggered = False
        self.transition_start_time = None
        self.transition_duration = 3.0  # 3 seconds pause

        #frame validatation checking
        self.is_user_in_frame = False
        self.frame_validation_count = 0
        self.required_validation_frames = 15  # 0.5 seconds at 30fps
        self.last_frame_status_sent = 0

    
    
    def send_data(self, data: dict):
        """✅ Send JSON data to Unity via data channel"""
        if self.data_channel and self.data_channel.readyState == "open":
            try:
                self.data_channel.send(json.dumps(data))
            except Exception as e:
                print(f"⚠️ Failed to send data: {e}")

    def check_user_in_frame(self, landmarks):
        """
        ✅ Check if user's full body is properly positioned in frame
        Returns: (is_in_frame: bool, feedback_message: str)
        """
        try:
            # Key landmarks for full body detection
            nose = landmarks[0]           # Head
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_ankle = landmarks[27]    # Feet
            right_ankle = landmarks[28]
            
            issues = []
            
            # ✅ 1. Check head visibility (should be in top 30% of frame)
            head_y = nose[1]
            if head_y < 0.05:
                issues.append("Move back - head too close to top")
            elif head_y > 0.3:
                issues.append("Move back - show your head properly")
            
            # ✅ 2. Check feet visibility (should be in bottom 20% of frame)
            avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            if avg_ankle_y < 0.75:
                issues.append("Move back - show your feet")
            elif avg_ankle_y > 0.95:
                issues.append("Move up - feet too close to bottom")

                # ✅ 3. Check horizontal centering (person should be centered)
            body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            if body_center_x < 0.25:
                issues.append("Move right - center yourself")
            elif body_center_x > 0.75:
                issues.append("Move left - center yourself")
            
            # ✅ 4. Check if person is not too close/far (shoulder width)
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            if shoulder_width < 0.1:
                issues.append("Move closer to camera")
            elif shoulder_width > 0.5:
                issues.append("Move back from camera")
            
            # ✅ 5. Check full body span (head to feet)
            body_height = avg_ankle_y - head_y
            if body_height < 0.4:
                issues.append("Show full body - move back")
            
            # ✅ 6. Check if all key landmarks are within frame boundaries
            all_landmarks = [nose, left_shoulder, right_shoulder, left_hip, right_hip, left_ankle, right_ankle]
            for i, (x, y) in enumerate(all_landmarks):
                if x < 0.05 or x > 0.95 or y < 0.05 or y > 0.95:
                    issues.append("Keep full body in frame")
                    break
            
            is_in_frame = len(issues) == 0
            message = "Perfect position! 👍" if is_in_frame else " • ".join(issues)
            
            return is_in_frame, message
            
        except (IndexError, TypeError) as e:
            return False, "Position yourself in front of camera"
    
    async def recv(self):
        """
        Process frame with MediaPipe and user positioning validation
        """
        # Get frame from incoming track
        frame = await self.track.recv()
        
        # Direct numpy array access
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Extract landmarks and send data
        if results.pose_landmarks:
            landmarks = [
                (lm.x, lm.y) for lm in results.pose_landmarks.landmark
            ]
            
            # ✅ STEP 1: Check if user is properly positioned
            is_positioned, positioning_feedback = self.check_user_in_frame(landmarks)
            
            # ✅ Track consistent positioning
            if is_positioned:
                self.frame_validation_count += 1
                if self.frame_validation_count >= self.required_validation_frames and not self.is_user_in_frame:
                    self.is_user_in_frame = True
                    self.send_data({
                        "type": "ready",
                        "message": "Perfect positioning! Starting detection...",
                        "ready": True
                    })
                    print("✅ User properly positioned - ready for detection")
            else:
                self.frame_validation_count = 0
                if self.is_user_in_frame:
                    self.is_user_in_frame = False
                    self.send_data({
                        "type": "user_not_ready", 
                        "message": positioning_feedback,
                        "ready": False
                    })
                    print(f"⚠️ User positioning lost: {positioning_feedback}")
            
            # ✅ Send positioning feedback every 15 frames (0.5 seconds)
            if self.frame_count % 15 == 0:
                progress = min(self.frame_validation_count / self.required_validation_frames * 100, 100)
                self.send_data({
                    "type": "frame_status",
                    "in_frame": is_positioned,
                    "message": positioning_feedback,
                    "readiness_progress": progress
                })
            
            # ✅ STEP 2: Only do detection if user is properly positioned
            if self.is_user_in_frame:
                if self.detector_type == "calibration":
                    if self.transition_triggered:
                        elapsed = time.time() - self.transition_start_time
                        remaining = self.transition_duration - elapsed
                        
                        if remaining > 0:
                            self.send_data({
                                "type": "calibration_transition_active",
                                "message": f"Turn to your RIGHT side ({remaining:.1f}s)",
                                "remaining": remaining,
                                "progress": self.detector.get_progress()["progress"]
                            })
                        else:
                            self.transition_triggered = False
                            self.send_data({
                                "type": "calibration_transition_complete",
                                "message": "Perfect! Hold the side pose",
                                "progress": self.detector.get_progress()["progress"]
                            })
                    else:
                        # ✅ Only add frames when user is positioned correctly
                        result = await self.detector.add_calibration_frame(landmarks, "general")
                        
                        self.send_data({
                            "type": "calibration_update",
                            "progress": result["progress"],
                            "message": result["message"],
                            "status": result["status"],
                            "phase": "front" if result["progress"] < 60 else "side",
                            "target": result.get("target", 100)
                        })
                        
                        if result["progress"] >= 60 and not self.transition_triggered:
                            self.transition_triggered = True
                            self.transition_start_time = time.time()
                            
                            self.send_data({
                                "type": "calibration_transition",
                                "message": "Great! Now turn to your RIGHT side",
                                "duration": self.transition_duration,
                                "progress": result["progress"]
                            })
                        
                        if result["status"] == "complete":
                            self.send_data({
                                "type": "calibration_complete",
                                "message": "Calibration complete! 🎉",
                                "calibration_id": self.detector.calibration_id,
                                "measurements": result.get("measurements", {})
                            })
                
                elif self.detector_type == "pushup":
                    # ✅ Only do pushup detection when user is positioned correctly
                    detection_result = self.detector.update(landmarks)
                    feedback = self.detector.get_feedback()
                    
                    self.send_data({
                        "type": "pushup_update",
                        "rep_count": detection_result['rep_count'],
                        "phase": detection_result["phase"],
                        "form_percentage": detection_result["form_percentage"],
                        "feedback": feedback,
                        "form_issues": detection_result["form_issues"],
                        "calibrated": detection_result["calibrated"],
                        "timestamp": self.frame_count
                    })
                    
                    if detection_result['rep_count'] > self.last_rep_count:
                        self.send_data({
                            "type": "rep_complete",
                            "rep_count": detection_result['rep_count'],
                            "form_issues": detection_result["form_issues"],
                            "good_form": len(detection_result["form_issues"]) == 0
                        })
                        self.last_rep_count = detection_result['rep_count']
            else:
                # ✅ User not positioned correctly - no detection
                pass  # Just wait for proper positioning
        
        else:
            # No pose detected - warn Unity
            self.send_data({
                "type": "warning",
                "message": "No pose detected - show full body in frame"
            })
        
        self.frame_count += 1
        
        # ✅ Return UNMODIFIED frame (Unity will draw overlays)
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        
        return new_frame

# Store active peer connections
pcs = set()

# ================================
# WEBRTC ENDPOINTS
# ================================

@router.post("/webrtc/offer")
async def webrtc_offer(request: dict):
    """
    Handle WebRTC offer from Unity
    Creates data channel for JSON communication
    """
    try:
        offer = RTCSessionDescription(
            sdp=request["sdp"],
            type=request["type"]
        )
        
        user_id = request.get("user_id")
        detector_type = request.get("detector_type", "pushup")
        session_id = request.get("session_id")  # For calibration sessions  
        
        print(f"📡 WebRTC offer received from user {user_id} for {detector_type}")
        
        # Create peer connection
        pc = RTCPeerConnection()
        pcs.add(pc)
        
        # ✅ Create data channel for sending detection data to Unity
        data_channel = pc.createDataChannel("detection_data")
        detection_track_ref = {"track": None}  # Store reference
        
        @data_channel.on("open")
        def on_data_channel_open():
            print(f"✅ Data channel opened for user {user_id}")
            # Send initial ready message
            data_channel.send(json.dumps({
                "type": "ready",
                "detector_type": detector_type,
                "message": f"{detector_type.title()} detection ready"
            }))
        
        @data_channel.on("message")
        def on_data_channel_message(message):
            """Handle commands from Unity (optional)"""
            try:
                data = json.loads(message)
                print(f"📨 Received from Unity: {data}")
                
                # Handle reset command
                if data.get("command") == "reset":
                    if detection_track_ref["track"]:
                        detection_track_ref["track"].detector.reset()
                        
                        # ✅ Reset transition state if calibration
                        if detection_track_ref["track"].detector_type == "calibration":
                            detection_track_ref["track"].transition_triggered = False
                            detection_track_ref["track"].transition_start_time = None
                        
                        data_channel.send(json.dumps({
                            "type": "reset_complete",
                            "message": "Detector reset"
                        }))
            except Exception as e:
                print(f"⚠️ Error processing Unity message: {e}")
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"🔗 Connection state changed to: {pc.connectionState}")
            
            if pc.connectionState == "connected":
                print(f"✅ Connection established for user {user_id}")
                # Signal ready to start collecting data
                if data_channel and data_channel.readyState == "open":
                    data_channel.send(json.dumps({
                        "type": "connection_ready",
                        "message": "WebRTC connected - starting detection",
                        "detector_type": detector_type
                    }))
            elif pc.connectionState == "failed":
                print(f"❌ Connection failed for user {user_id}")
                await pc.close()
                pcs.discard(pc)
            elif pc.connectionState == "closed":
                print(f"🔌 Connection closed for user {user_id}")
                pcs.discard(pc)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"🧊 ICE Connection State: {pc.iceConnectionState}")
            if pc.iceConnectionState == "connected" and pc.iceGatheringState == "complete":
                print("🎯 ICE Connection ready for data transfer")
                if data_channel and data_channel.readyState == "open":
                    data_channel.send(json.dumps({
                        "type": "ice_ready",
                        "message": "ICE connection established"
                    }))

        # Add negotiation needed handler
        @pc.on("negotiationneeded")
        async def on_negotiationneeded():
            print("📝 Negotiation needed event fired")
        
        @pc.on("track")
        async def on_track(track):
            print(f"✅ Track {track.kind} received from user {user_id}")
            
            if track.kind == "video":
                # ✅ Create detection track with data channel
                detection_track = DetectionVideoTrack(
                    track=track,
                    detector_type=detector_type,
                    user_id=user_id,
                    data_channel=data_channel  # Pass data channel
                    ,session_id=session_id
                )
                
                detection_track_ref["track"] = detection_track  # Store reference
                
                pc.addTrack(detection_track)
                print(f"🎥 Detection track added for user {user_id}")
        
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        print(f"✅ SDP answer created for user {user_id}")
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ WebRTC offer error: {e}")
        raise HTTPException(status_code=500, detail=f"WebRTC setup failed: {str(e)}")

# ================================
# REST ENDPOINTS (unchanged)
# ================================
def _gc_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in active_calibration_sessions.items() if s.get("expires_at", 0) <= now]
    for sid in expired:
        active_calibration_sessions.pop(sid, None)


@router.post("/calibration-start")
async def start_calibration(user_data: Dict[str, Any] = Depends(require_firebase_user),
                            exercise: Optional[str] = None):
    """
    Create a calibration session. Unity should call this with:
      Authorization: Bearer <Firebase ID token>
    Optionally pass `exercise` as a query/body param if you want to tag the session.
    """
    try:
        _gc_sessions()
        user_id = user_data["uid"]

        calibrator = BodyCalibrator(
            user_id=user_id,
            firebase_client=firebase_service.db
        )

        session_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        active_calibration_sessions[session_id] = {
            "calibrator": calibrator,
            "user_id": user_id,
            "created_at": time.time(),
            "expires_at": time.time() + SESSION_TTL,
            "status": "waiting_for_webrtc",
            "exercise": exercise,
        }

        return {
            "status": "success",
            "message": "Calibration session created - connect via WebRTC",
            "session_id": session_id,
            "calibration_id": calibrator.calibration_id,
            "user_id": user_id,
            "expires_in": SESSION_TTL
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calibration/status/{user_id}")
async def get_calibration_status(user_id: str):
    """Check if user has completed calibration"""
    try:
        calibrator = BodyCalibrator(
            user_id=user_id,
            firebase_client=firebase_service.db
        )
        
        is_calibrated = calibrator.load_user_calibration()
        
        if is_calibrated:
            return {
                "status": "success",
                "calibrated": True,
                "calibration_id": calibrator.calibration_id,
                "measurements": calibrator.baseline_measurements
            }
        else:
            return {
                "status": "success",
                "calibrated": False,
                "message": "No calibration found for user"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/pushup/start")
# async def start_pushup_session(user_data: dict = Depends(get_user_from_auth)):
#     """Start a new push-up detection session"""
#     try:
#         user_id = user_data.get("uid")
        
#         calibrator = BodyCalibrator(
#             user_id=user_id,
#             firebase_client=firebase_service.db
#         )
        
#         is_calibrated = calibrator.load_user_calibration()
        
#         return {
#             "status": "success",
#             "message": "Push-up session ready - connect via WebRTC",
#             "user_id": user_id,
#             "calibrated": is_calibrated,
#             "warning": None if is_calibrated else "User not calibrated - using generic thresholds"
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/pushup/history/{user_id}")
# async def get_pushup_history(user_id: str, limit: int = 10):
#     """Get user's push-up workout history"""
#     try:
#         # TODO: Implement workout history storage/retrieval
#         return {
#             "status": "success",
#             "message": "History endpoint - to be implemented",
#             "user_id": user_id
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Check if detection services are running"""
    return {
        "status": "healthy",
        "active_webrtc_connections": len(pcs),
        "mediapipe_available": True,
        "firebase_connected": firebase_service.db is not None
    }

            
@router.on_event("shutdown")
async def on_shutdown():
    """Close all peer connections on shutdown"""
    print("🔄 Closing all WebRTC connections...")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    print("✅ All WebRTC connections closed")