from app.services.mediapipe.base_detector import BasePoseDetector
import math
import time
import cv2


#  Beyond calibration: ML-based classifiers

# If you want even higher robustness, you can:

# Collect labeled data (videos of correct vs incorrect squats).
# Extract features (angles, velocities, joint‐distance ratios) per frame.
# Train a small classifier (e.g. SVM or a lightweight neural network) that ingests those features and outputs “good squat” vs “not.”
# That approach is more work but can capture subtle form errors (knees caving in, back rounding, etc.) that simple angle‐thresholds won’t.

class SquatDetector(BasePoseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.standing_angle =None
        self.bottom_angle =None
        self.cutoff = 0.8 


    def calibrate(
        self,
        cap: "cv2.VideoCapture",
        pose: str = "standing",
        samples: int = 30,
        max_attempts: int = 300
    ):
        """
        Calibrate either the 'standing' or 'bottom' angle.
        cap: an open cv2.VideoCapture
        pose: 'standing' or 'bottom'
        samples: how many *valid* readings you want
        max_attempts: give up after this many frames
        """
        collected = []
        attempts  = 0

        print(f"📝 Calibrating {pose} pose: please hold still in view...")
        while len(collected) < samples and attempts < max_attempts:
            attempts += 1
            ret, frame = cap.read()
            if not ret:
                continue

            lm = self._process(frame)
            if not lm:
                # no person detected this frame
                continue

            # extract left hip/knee/ankle
            a = lm.landmark
            hip   = a[self.mp_pose.PoseLandmark.LEFT_HIP]
            knee  = a[self.mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = a[self.mp_pose.PoseLandmark.LEFT_ANKLE]

            angle = self.calculate_angle(hip, knee, ankle)
            collected.append(angle)

            # optional: show progress
            if attempts % 30 == 0:
                print(f"  → {len(collected)}/{samples} valid samples so far...")

            # small sleep so user can adjust
            time.sleep(0.05)

        if not collected:
            raise RuntimeError(f"❌ Calibration failed for '{pose}' — no valid frames detected.")

        avg_angle = sum(collected) / len(collected)
        if pose == "standing":
            self.standing_angle = avg_angle
        else:
            self.bottom_angle = avg_angle

        print(f"✅ {pose.capitalize()} angle: {avg_angle:.1f}° "
              f"({len(collected)} samples over {attempts} attempts)")
        return avg_angle

    def calculate_angle(self, a, b, c):
        # a, b, c are (x,y,z) tuples; returns angle at b
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        cos_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (
             math.hypot(*ba) * math.hypot(*bc) + 1e-6
        )
        return math.degrees(math.acos(max(min(cos_angle,1), -1)))

    def detect(self, frame):
        if self.standing_angle is None or self.bottom_angle is None:
            return {"error": "not calibrated"}

        landmarks = self._process(frame)
        if not landmarks:
            return {"error": "no person detected"}
        # hip, knee, ankle
        hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        angle = self.calculate_angle(hip, knee, ankle)
        # is_squat = angle < 90  # simple threshold


        depth = (self.standing_angle - angle) / (self.standing_angle - self.bottom_angle + 1e-6)
        is_squat = depth >= self.cutoff
        return {"knee_angle": angle, "depth": depth, "is_squat": is_squat}
