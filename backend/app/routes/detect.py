from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
import base64
import numpy as np
import cv2

from app.services.mediapipe.base_detector import BasePoseDetector
from app.services.classifier.classifier import predict_from_landmarks


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
                await ws.send_json({"label": "error", "msg": "invalid frame"})
                continue

            landmarks = pose_detector._process(frame)
            if not landmarks:
                await ws.send_json({"label": "none"})
                continue

            coords = [(p.x, p.y) for p in landmarks.landmark]
            label = predict_from_landmarks(coords)

            await ws.send_json({"label": label})
    except Exception as e:
        print("WebSocket error:", e)
        await ws.close()
    