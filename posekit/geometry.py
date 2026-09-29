"""Joint angles and rule-based posture classification from 2D keypoints."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .skeleton import KP

MIN_CONF = 0.4


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees (at vertex b)."""
    ba, bc = a - b, c - b
    cos = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _pt(kpts: np.ndarray, name: str) -> np.ndarray | None:
    x, y, c = kpts[KP[name]]
    return np.array([x, y]) if c >= MIN_CONF else None


def _mean(*pts):
    pts = [p for p in pts if p is not None]
    return np.mean(pts, axis=0) if pts else None


def joint_angles(kpts: np.ndarray) -> dict[str, float | None]:
    """Elbow, knee and hip angles for both sides (None when a keypoint is not visible)."""
    out: dict[str, float | None] = {}
    for side in ("left", "right"):
        triples = {
            f"{side}_elbow": (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"),
            f"{side}_knee": (f"{side}_hip", f"{side}_knee", f"{side}_ankle"),
            f"{side}_hip": (f"{side}_shoulder", f"{side}_hip", f"{side}_knee"),
        }
        for name, (a, b, c) in triples.items():
            pa, pb, pc = _pt(kpts, a), _pt(kpts, b), _pt(kpts, c)
            out[name] = round(angle(pa, pb, pc), 1) if all(p is not None for p in (pa, pb, pc)) else None
    return out


def torso_inclination(kpts: np.ndarray) -> float | None:
    """Angle between the shoulders->hips segment and the vertical (0 = upright, 90 = horizontal)."""
    sh = _mean(_pt(kpts, "left_shoulder"), _pt(kpts, "right_shoulder"))
    hp = _mean(_pt(kpts, "left_hip"), _pt(kpts, "right_hip"))
    if sh is None or hp is None:
        return None
    dx, dy = hp - sh
    return round(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9)), 1)


def thigh_torso_ratio(kpts: np.ndarray) -> float | None:
    """Projected thigh length / torso length (about 0.8-1.1 standing, much lower when seated facing the camera)."""
    sh = _mean(_pt(kpts, "left_shoulder"), _pt(kpts, "right_shoulder"))
    hp = _mean(_pt(kpts, "left_hip"), _pt(kpts, "right_hip"))
    thighs = [np.linalg.norm(k - h) for h, k in ((_pt(kpts, "left_hip"), _pt(kpts, "left_knee")),
                                                  (_pt(kpts, "right_hip"), _pt(kpts, "right_knee")))
              if h is not None and k is not None]
    if sh is None or hp is None or not thighs:
        return None
    torso = np.linalg.norm(sh - hp)
    return round(float(np.mean(thighs) / torso), 2) if torso > 1 else None


@dataclass
class Posture:
    label: str          # standing | sitting | lying | unknown
    arms_raised: bool
    torso_deg: float | None
    knee_deg: float | None
    hip_deg: float | None


def classify_posture(kpts: np.ndarray) -> Posture:
    ang = joint_angles(kpts)
    torso = torso_inclination(kpts)
    knees = [v for k, v in ang.items() if k.endswith("knee") and v is not None]
    hips = [v for k, v in ang.items() if k.endswith("hip") and v is not None]
    knee = float(np.mean(knees)) if knees else None
    hip = float(np.mean(hips)) if hips else None

    thigh = thigh_torso_ratio(kpts)
    if torso is None:
        label = "unknown"
    elif torso > 60:
        label = "lying"
    elif hip is not None and hip < 125 and (knee is None or knee < 135):
        label = "sitting"
    elif thigh is not None and thigh < 0.45:
        # seen from the front, a seated person's thighs point at the camera: the hip angle
        # stays open in 2D but the thighs look very short compared with the torso
        label = "sitting"
    elif hip is not None or knee is not None:
        label = "standing"
    else:
        label = "unknown"   # lower body not visible (e.g. portrait crop)

    wrists = [_pt(kpts, "left_wrist"), _pt(kpts, "right_wrist")]
    sh = _mean(_pt(kpts, "left_shoulder"), _pt(kpts, "right_shoulder"))
    arms_raised = sh is not None and any(w is not None and w[1] < sh[1] for w in wrists)
    return Posture(label, arms_raised, torso, None if knee is None else round(knee, 1),
                   None if hip is None else round(hip, 1))
