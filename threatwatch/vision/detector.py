"""
YOLOv8 Pose Estimation Detector.
Loads lightweight YOLOv8-pose model with CUDA acceleration for real-time person & keypoint detection.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO

from threatwatch.config import POSE_CONF_THRESHOLD, YOLO_MODEL_NAME


class PoseDetector:
    """
    Wrapper for YOLOv8n-pose real-time inference.
    """

    # COCO Keypoint index definitions
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # Skeleton connection pairs for drawing
    SKELETON_PAIRS = [
        (5, 6),   # shoulders
        (5, 7), (7, 9),   # left arm
        (6, 8), (8, 10),  # right arm
        (5, 11), (6, 12), # torso
        (11, 12),         # hips
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16)  # right leg
    ]

    def __init__(self, model_name: str = YOLO_MODEL_NAME):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[PoseDetector] Initializing YOLOv8-pose model ({model_name}) on {self.device}...")
        try:
            self.model = YOLO(model_name)
            # Warmup inference
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
            print("[PoseDetector] YOLOv8-pose successfully initialized.")
        except Exception as e:
            print(f"[PoseDetector] Warning: Could not initialize YOLOv8 ({e}). Falling back to CPU.")
            self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Run pose inference on image frame.
        Returns detection dictionary for the primary person (largest bbox), or None.
        """
        if frame is None or self.model is None:
            return None

        results = self.model.predict(
            frame,
            conf=POSE_CONF_THRESHOLD,
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return None

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return None

        # Find person with largest bounding box area
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts_data = res.keypoints.data.cpu().numpy()  # (N, 17, 3) -> [x, y, conf]

        max_area = -1
        best_idx = -1
        for i, bbox in enumerate(boxes):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > max_area:
                max_area = area
                best_idx = i

        if best_idx == -1:
            return None

        bbox = boxes[best_idx].tolist()
        conf = float(confs[best_idx])
        kpts = kpts_data[best_idx]  # Shape (17, 3)

        return {
            "bbox": bbox,  # [x1, y1, x2, y2]
            "confidence": conf,
            "keypoints": kpts,  # (17, 3)
            "total_persons_detected": len(boxes),
        }
