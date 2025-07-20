from app.services.mediapipe.base_detector import BasePoseDetector
from app.services.mediapipe.squat_detector import SquatDetector  # reuse angle func
class PlankDetector(BasePoseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def detect(self, frame):
        landmarks = self._process(frame)
        if not landmarks:
            return {"error": "no person detected"}
        # check shoulder-hip-ankle alignment: small angle ≈ straight line
        s = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        h = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        a = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        sd = SquatDetector()
        angle = sd.calculate_angle(s, h, a)
        is_plank = abs(angle - 180) < 10
        return {"spine_angle": angle, "is_plank": is_plank}
