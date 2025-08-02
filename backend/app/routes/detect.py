from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from datetime import datetime
from app.services.mediapipe.base_detector import BasePoseDetector
from app.services.classifier.classifier import analyze_pose

pose_detector = BasePoseDetector(static_image_mode=False, min_detection_confidence=0.7)
router = APIRouter()

class ImageInput(BaseModel):
    image: str  # base64 string

@router.websocket("/pose_detection")
async def stream_pose_estimation(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()  # base64 string
            img_data = base64.b64decode(data)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await ws.send_json({
                    "error": "invalid_frame",
                    "message": "Could not decode frame"
                })
                continue
            
            landmarks = pose_detector._process(frame)
            
            if not landmarks:
                await ws.send_json({
                    "exercise": "none",
                    "exercise_confidence": 0.0,
                    "form_quality": None,
                    "form_confidence": 0.0,
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Extract coordinates and run hybrid analysis
            coords = [(p.x, p.y) for p in landmarks.landmark]
            exercise, exercise_confidence, form_quality, form_confidence = analyze_pose(coords)
            
            # Generate feedback based on results
            feedback = generate_feedback(exercise, form_quality, form_confidence)
            
            # Send comprehensive results
            await ws.send_json({
                "exercise": exercise,
                "exercise_confidence": float(exercise_confidence),
                "form_quality": form_quality,
                "form_confidence": float(form_confidence) if form_confidence else 0.0,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print("WebSocket error:", e)
        try:
            await ws.send_json({
                "error": "processing_error", 
                "message": str(e)
            })
        except:
            pass
        await ws.close()

def generate_feedback(exercise: str, form_quality: str, form_confidence: float):
    """Generate coaching feedback based on analysis results"""
    if not form_quality or form_confidence < 0.5:
        return {
            "message": "Analyzing your form...",
            "type": "analyzing",
            "corrections": []
        }
    
    feedback_map = {
        "excellent": {
            "message": "Perfect form! 🏆 Keep it up!",
            "type": "success",
            "corrections": []
        },
        "good": {
            "message": "Good form! Minor adjustments needed",
            "type": "warning", 
            "corrections": get_exercise_tips(exercise, "good")
        },
        "poor": {
            "message": "Focus on your technique",
            "type": "error",
            "corrections": get_exercise_tips(exercise, "poor")
        }
    }
    
    return feedback_map.get(form_quality, {
        "message": "Keep working on your form",
        "type": "info",
        "corrections": []
    })

def get_exercise_tips(exercise: str, quality: str):
    """Exercise-specific coaching tips"""
    tips = {
        "squat": {
            "good": ["Keep knees aligned with toes", "Maintain straight back"],
            "poor": ["Lower down further", "Don't let knees cave in", "Keep chest up"]
        },
        "pushup": {
            "good": ["Maintain straight line from head to heels", "Control the descent"],
            "poor": ["Don't let hips sag", "Go down to chest level", "Keep core tight"]
        },
        "plank": {
            "good": ["Hold the position steady", "Keep breathing"],
            "poor": ["Don't lift hips too high", "Engage your core", "Keep straight line"]
        }
    }
    
    return tips.get(exercise, {}).get(quality, ["Focus on proper technique"])