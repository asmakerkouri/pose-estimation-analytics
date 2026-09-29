"""Skeleton rendering with OpenCV."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import MIN_CONF
from .skeleton import EDGES, KP

# BGR, colour-blind-safe categorical order
COLORS = {"head": (214, 120, 42), "arms": (52, 104, 235), "torso": (122, 175, 27), "legs": (0, 161, 237)}


def draw_people(image: np.ndarray, people, show_label: bool = True) -> np.ndarray:
    out = image.copy()
    scale = max(1, round(max(out.shape[:2]) / 640))
    for p in people:
        k = p.keypoints
        for a, b, group in EDGES:
            (xa, ya, ca), (xb, yb, cb) = k[KP[a]], k[KP[b]]
            if ca >= MIN_CONF and cb >= MIN_CONF:
                cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), COLORS[group], 3 * scale, cv2.LINE_AA)
        for x, y, c in k:
            if c >= MIN_CONF:
                cv2.circle(out, (int(x), int(y)), 4 * scale, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, (int(x), int(y)), 4 * scale, (40, 40, 40), 1, cv2.LINE_AA)
        if show_label and p.posture.label != "unknown":
            x1, y1 = int(p.box[0]), int(p.box[1])
            text = p.posture.label + (" | arms up" if p.posture.arms_raised else "")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, scale)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, max(th + 8, y1)), (20, 20, 20), -1)
            cv2.putText(out, text, (x1 + 4, max(th + 4, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale,
                        (255, 255, 255), scale, cv2.LINE_AA)
    return out
