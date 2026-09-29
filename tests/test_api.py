import sys
from pathlib import Path

import cv2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("ultralytics")
from fastapi.testclient import TestClient  # noqa: E402

from posekit.api import app  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "bus.jpg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_pose_json(client):
    r = client.post("/pose", files={"file": ("bus.jpg", SAMPLE.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 3
    assert len(body["people"][0]["keypoints"]) == 17
    assert "standing" in {p["posture"] for p in body["people"]}


def test_pose_image(client):
    r = client.post("/pose/image", files={"file": ("bus.jpg", SAMPLE.read_bytes(), "image/jpeg")})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert cv2.imdecode(__import__("numpy").frombuffer(r.content, "uint8"), 1) is not None


def test_rejects_non_image(client):
    assert client.post("/pose", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
