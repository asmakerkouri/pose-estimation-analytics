"""YOLO11-pose wrapper (PyTorch or ONNX Runtime back-end through Ultralytics)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .geometry import Posture, classify_posture, joint_angles


@dataclass
class Person:
    box: list[float]
    score: float
    keypoints: np.ndarray    # (17, 3): x, y, confidence
    posture: Posture
    angles: dict

    def to_dict(self) -> dict:
        return {"box": [round(v, 1) for v in self.box], "score": round(self.score, 3),
                "posture": self.posture.label, "arms_raised": self.posture.arms_raised,
                "torso_inclination_deg": self.posture.torso_deg, "angles": self.angles,
                "keypoints": np.round(self.keypoints, 1).tolist()}


class PoseDetector:
    def __init__(self, weights: str | None = None, conf: float = 0.35, imgsz: int = 640):
        from ultralytics import YOLO

        self.weights = weights or os.getenv("POSE_WEIGHTS", "yolo11n-pose.pt")
        self.model = YOLO(self.weights, task="pose")
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, image) -> list[Person]:
        r = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return []
        kps = r.keypoints.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        return [Person(b.tolist(), float(s), k, classify_posture(k), joint_angles(k))
                for b, s, k in zip(boxes, scores, kps, strict=True)]
