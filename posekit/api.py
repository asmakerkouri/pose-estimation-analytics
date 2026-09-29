"""REST API: POST an image, get skeletons, joint angles and postures (JSON or annotated JPEG)."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from .detector import PoseDetector
from .draw import draw_people

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["detector"] = PoseDetector()
    yield
    state.clear()


app = FastAPI(title="Pose estimation API", version="1.0.0", lifespan=lifespan)


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "File is not a valid image")
    return img


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "weights": getattr(state.get("detector"), "weights", None)}


@app.post("/pose")
async def pose(file: UploadFile = File(...)) -> dict:
    img = _decode(await file.read())
    t0 = time.perf_counter()
    people = state["detector"](img)
    return {"count": len(people), "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "people": [p.to_dict() for p in people]}


@app.post("/pose/image")
async def pose_image(file: UploadFile = File(...)) -> Response:
    img = _decode(await file.read())
    ok, buf = cv2.imencode(".jpg", draw_people(img, state["detector"](img)))
    return Response(buf.tobytes(), media_type="image/jpeg")
