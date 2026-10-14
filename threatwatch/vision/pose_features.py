"""
Computer Vision Pose & Facial Expression Analyzer.
Detects upright posture, close-up framing, defensive guards, falls,
and facial expressions (smiling, neutral, distressed).
"""

import time
import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

from threatwatch.vision.detector import PoseDetector


class PoseFeatureAnalyzer:
    """
    Analyzes body pose, motion, and facial expressions (including smiles).
    """

    def __init__(self):
        self._prev_keypoints: Optional[np.ndarray] = None
        self._prev_time = time.time()
        self._motion_history: List[float] = []
        
        # Load OpenCV Smile Cascade for expression detection
        try:
            self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        except Exception:
            self.smile_cascade = None

    def analyze(self, detection: Optional[Dict[str, Any]], frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract posture, facial expression (smile/neutral/distress), velocity, and visual distress.
        """
        now = time.time()
        dt = max(now - self._prev_time, 0.001)
        self._prev_time = now
        h, w = frame.shape[:2]

        if detection is None:
            self._prev_keypoints = None
            return {
                "person_detected": False,
                "posture": "No Person",
                "facial_expression": "None",
                "is_smiling": False,
                "movement_level": "None",
                "motion_velocity": 0.0,
                "visual_distress_score": 0.0,
                "confidence": 0.0,
                "is_fallen": False,
                "is_defensive": False,
                "is_closeup": False,
            }

        kpts = detection["keypoints"]  # (17, 3) [x, y, conf]
        bbox = detection["bbox"]  # [x1, y1, x2, y2]
        conf = float(detection["confidence"])

        nose = kpts[0]
        l_eye, r_eye = kpts[1], kpts[2]
        l_ear, r_ear = kpts[3], kpts[4]
        l_sh, r_sh = kpts[5], kpts[6]
        l_wr, r_wr = kpts[9], kpts[10]
        l_hip, r_hip = kpts[11], kpts[12]
        l_ank, r_ank = kpts[15], kpts[16]

        bbox_w = max(bbox[2] - bbox[0], 1.0)
        bbox_h = max(bbox[3] - bbox[1], 1.0)

        # 1. Motion Velocity calculation
        motion_vel = 0.0
        if self._prev_keypoints is not None:
            valid_mask = (kpts[:, 2] > 0.4) & (self._prev_keypoints[:, 2] > 0.4)
            if np.sum(valid_mask) >= 3:
                diffs = np.linalg.norm(kpts[valid_mask, :2] - self._prev_keypoints[valid_mask, :2], axis=1)
                diag = math.sqrt(h**2 + w**2)
                raw_vel = float(np.mean(diffs) / (diag * dt)) * 10.0
                motion_vel = max(0.0, raw_vel - 0.20)

        self._prev_keypoints = kpts.copy()
        self._motion_history.append(motion_vel)
        if len(self._motion_history) > 8:
            self._motion_history.pop(0)
        smooth_motion = float(np.mean(self._motion_history))

        # 2. Face & Close-Up Detection
        eyes_detected = l_eye[2] > 0.35 and r_eye[2] > 0.35
        head_detected = nose[2] > 0.35 or eyes_detected
        shoulders_detected = l_sh[2] > 0.35 and r_sh[2] > 0.35

        inter_eye_dist = 0.0
        if eyes_detected:
            inter_eye_dist = np.linalg.norm(r_eye[:2] - l_eye[:2])

        is_closeup = (inter_eye_dist > 55.0) or (bbox_w > w * 0.55)

        # 3. Facial Expression & Smile Detection
        is_smiling = False
        facial_expression = "Neutral"
        expression_distress = 0.0

        # Run smile detection on face region if face is visible
        if head_detected:
            try:
                # Estimate face bounding box
                if eyes_detected and nose[2] > 0.35:
                    fx_center = int(nose[0])
                    fy_center = int((nose[1] + (l_eye[1] + r_eye[1]) / 2) / 2)
                    fw = int(max(inter_eye_dist * 2.2, 50))
                    fh = int(fw * 1.2)
                else:
                    fx_center = int((bbox[0] + bbox[2]) / 2)
                    fy_center = int(bbox[1] + bbox_h * 0.25)
                    fw = int(bbox_w * 0.6)
                    fh = int(fw * 1.2)

                x1_f = max(0, fx_center - fw // 2)
                y1_f = max(0, fy_center - fh // 2)
                x2_f = min(w, fx_center + fw // 2)
                y2_f = min(h, fy_center + fh // 2)

                if (x2_f - x1_f) > 30 and (y2_f - y1_f) > 30:
                    face_roi = frame[y1_f:y2_f, x1_f:x2_f]
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Mouth region (lower 50% of face)
                    mouth_roi = gray_face[int(gray_face.shape[0] * 0.5):, :]
                    
                    if self.smile_cascade and mouth_roi.shape[0] > 10 and mouth_roi.shape[1] > 10:
                        smiles = self.smile_cascade.detectMultiScale(
                            mouth_roi,
                            scaleFactor=1.4,
                            minNeighbors=15,
                            minSize=(15, 15),
                        )
                        if len(smiles) > 0:
                            is_smiling = True

                # Determine facial expression
                if is_smiling:
                    facial_expression = "Smiling"
                    expression_distress = -0.03  # Smiling reduces distress
                elif smooth_motion > 2.0:
                    facial_expression = "Agitated"
                    expression_distress = 0.30
                else:
                    facial_expression = "Neutral"
                    expression_distress = 0.0

            except Exception:
                facial_expression = "Neutral"

        # 4. Fall / Horizontal Collapse Detection
        is_fallen = False
        if shoulders_detected:
            sdx = r_sh[0] - l_sh[0]
            sdy = r_sh[1] - l_sh[1]
            shoulder_tilt_deg = abs(math.degrees(math.atan2(sdy, sdx)))
            if shoulder_tilt_deg > 90:
                shoulder_tilt_deg = 180 - shoulder_tilt_deg
            if shoulder_tilt_deg > 50.0:
                is_fallen = True

        # Check full body collapse if hips & ankles visible
        if l_hip[2] > 0.35 and r_hip[2] > 0.35 and l_ank[2] > 0.35 and r_ank[2] > 0.35:
            body_h = abs(((l_ank[1] + r_ank[1]) / 2) - ((l_sh[1] + r_sh[1]) / 2))
            body_w = abs(((l_ank[0] + r_ank[0]) / 2) - ((l_sh[0] + r_sh[0]) / 2))
            if body_h < body_w * 0.65:
                is_fallen = True

        # 5. Defensive Posture Detection
        is_defensive = False
        if head_detected and (l_wr[2] > 0.45 or r_wr[2] > 0.45):
            head_y = nose[1] if nose[2] > 0.35 else (l_eye[1] if eyes_detected else 0)
            head_x = nose[0] if nose[2] > 0.35 else (w / 2)

            if l_wr[2] > 0.45 and l_wr[1] <= (head_y + 35) and abs(l_wr[0] - head_x) < (w * 0.20):
                is_defensive = True
            if r_wr[2] > 0.45 and r_wr[1] <= (head_y + 35) and abs(r_wr[0] - head_x) < (w * 0.20):
                is_defensive = True

        # 6. Posture Classification (Clean, Simple Strings)
        if is_fallen:
            posture = "Fallen / Lying Down"
        elif is_defensive:
            posture = "Defensive Guard"
        elif smooth_motion > 2.2:
            posture = "Rapid Struggle"
        elif is_closeup:
            posture = "Sitting (Close-up)"
        elif shoulders_detected:
            posture = "Sitting Upright"
        else:
            posture = "Upper Body"

        # Movement Level
        if smooth_motion > 2.2:
            movement_level = "High Motion"
        elif smooth_motion > 0.9:
            movement_level = "Moderate"
        else:
            movement_level = "Normal"

        # 7. Calibrated Visual Distress Score [0.0 - 1.0]
        v_score = 0.05 + expression_distress

        if is_fallen:
            v_score += 0.75
        elif is_defensive:
            v_score += 0.65

        if smooth_motion > 2.2:
            v_score += 0.25
        elif smooth_motion > 0.9:
            v_score += 0.08

        v_score = float(np.clip(v_score, 0.02, 1.0))

        return {
            "person_detected": True,
            "posture": str(posture),
            "facial_expression": str(facial_expression),
            "is_smiling": bool(is_smiling),
            "movement_level": str(movement_level),
            "motion_velocity": float(round(smooth_motion, 2)),
            "visual_distress_score": float(round(v_score, 3)),
            "confidence": float(round(conf, 2)),
            "is_fallen": bool(is_fallen),
            "is_defensive": bool(is_defensive),
            "is_closeup": bool(is_closeup),
        }

    def render_overlay(self, frame: np.ndarray, detection: Optional[Dict[str, Any]], analysis: Dict[str, Any]) -> np.ndarray:
        """
        Draw clean, professional HUD overlay on camera frame.
        """
        out_frame = frame.copy()
        h, w, _ = out_frame.shape

        color_cyan = (255, 240, 0)   # BGR for #00f0ff
        color_neon = (0, 255, 204)   # BGR for #ccff00
        color_alert = (60, 0, 255)   # BGR for #ff003c
        color_dark = (17, 17, 17)

        if detection is not None and analysis["person_detected"]:
            bbox = [int(v) for v in detection["bbox"]]
            kpts = detection["keypoints"]
            x1, y1, x2, y2 = bbox

            v_score = analysis["visual_distress_score"]
            accent_color = color_neon if v_score < 0.35 else (color_cyan if v_score < 0.65 else color_alert)

            # Clean bounding box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), accent_color, 1)

            # Draw Pose Skeleton
            for p1_idx, p2_idx in PoseDetector.SKELETON_PAIRS:
                kp1, kp2 = kpts[p1_idx], kpts[p2_idx]
                if kp1[2] > 0.35 and kp2[2] > 0.35:
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(out_frame, pt1, pt2, (0, 240, 255), 2)

            # Draw keypoint dots
            for kp in kpts:
                if kp[2] > 0.35:
                    pt = (int(kp[0]), int(kp[1]))
                    cv2.circle(out_frame, pt, 3, accent_color, -1)

        # Top Status Bar
        hud_h = 30
        overlay = out_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), color_dark, -1)
        cv2.addWeighted(overlay, 0.85, out_frame, 0.15, 0, out_frame)
        cv2.line(out_frame, (0, hud_h), (w, hud_h), (45, 45, 45), 1)

        expr_str = analysis.get("facial_expression", "Neutral")
        posture_str = analysis.get("posture", "Sitting")
        distress_pct = int(analysis.get("visual_distress_score", 0.05) * 100)
        
        status_text = f"{posture_str} | Expression: {expr_str} | Distress: {distress_pct}%"
        cv2.putText(out_frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

        return out_frame
