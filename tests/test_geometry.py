import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posekit.geometry import angle, classify_posture, joint_angles, torso_inclination  # noqa: E402
from posekit.reps import RepCounter  # noqa: E402
from posekit.skeleton import EDGES, KEYPOINTS, KP  # noqa: E402


def person(**pts):
    """17x3 keypoint array; unspecified keypoints are invisible (confidence 0)."""
    k = np.zeros((17, 3))
    for name, (x, y) in pts.items():
        k[KP[name]] = (x, y, 0.9)
    return k


STANDING = dict(left_shoulder=(90, 100), right_shoulder=(110, 100), left_elbow=(88, 150), right_elbow=(112, 150),
                left_wrist=(87, 195), right_wrist=(113, 195), left_hip=(93, 200), right_hip=(107, 200),
                left_knee=(93, 280), right_knee=(107, 280), left_ankle=(93, 360), right_ankle=(107, 360))


def test_skeleton_definition():
    assert len(KEYPOINTS) == 17
    assert all(a in KP and b in KP for a, b, _ in EDGES)


def test_angle():
    assert angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])) == pytest.approx(90)
    assert angle(np.array([-1, 0]), np.array([0, 0]), np.array([1, 0])) == pytest.approx(180, abs=0.01)


def test_standing():
    k = person(**STANDING)
    p = classify_posture(k)
    assert p.label == "standing" and not p.arms_raised
    assert joint_angles(k)["left_knee"] == pytest.approx(180, abs=1)
    assert torso_inclination(k) == pytest.approx(0, abs=1)


def test_sitting_side_view():
    k = person(left_shoulder=(100, 100), left_hip=(100, 200), left_knee=(180, 200), left_ankle=(180, 280))
    assert classify_posture(k).label == "sitting"


def test_lying():
    k = person(left_shoulder=(100, 300), right_shoulder=(100, 320), left_hip=(200, 305), right_hip=(200, 318))
    assert classify_posture(k).label == "lying"


def test_arms_raised():
    pts = dict(STANDING, left_wrist=(80, 40))
    assert classify_posture(person(**pts)).arms_raised


def test_invisible_keypoints():
    assert classify_posture(np.zeros((17, 3))).label == "unknown"
    assert all(v is None for v in joint_angles(np.zeros((17, 3))).values())


def test_rep_counter_counts_full_cycles_only():
    c = RepCounter(smooth=1)
    for a in [170, 150, 90, 120, 165, 170, 95, 150, None, 170, 140, 110]:  # 2 full squats, 1 partial
        c.update(a)
    assert c.reps == 2
