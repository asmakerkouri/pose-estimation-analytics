"""COCO-17 keypoint layout and skeleton edges."""
from __future__ import annotations

KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINTS)}

# (a, b, group) - group drives the colour: head, arms, torso, legs
EDGES = [
    ("left_eye", "right_eye", "head"), ("nose", "left_eye", "head"), ("nose", "right_eye", "head"),
    ("left_eye", "left_ear", "head"), ("right_eye", "right_ear", "head"),
    ("left_shoulder", "right_shoulder", "torso"), ("left_shoulder", "left_hip", "torso"),
    ("right_shoulder", "right_hip", "torso"), ("left_hip", "right_hip", "torso"),
    ("left_shoulder", "left_elbow", "arms"), ("left_elbow", "left_wrist", "arms"),
    ("right_shoulder", "right_elbow", "arms"), ("right_elbow", "right_wrist", "arms"),
    ("left_hip", "left_knee", "legs"), ("left_knee", "left_ankle", "legs"),
    ("right_hip", "right_knee", "legs"), ("right_knee", "right_ankle", "legs"),
]
