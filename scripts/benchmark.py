"""CPU latency: PyTorch vs ONNX Runtime for the same YOLO11n-pose model.

python scripts/benchmark.py  -> docs/images/benchmark.json
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def bench(model, img, n=30, warmup=5):
    for _ in range(warmup):
        model.predict(img, imgsz=640, verbose=False)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(img, imgsz=640, verbose=False)
        t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(statistics.median(t), 1), "p95_ms": round(sorted(t)[int(0.95 * n) - 1], 1)}


def main() -> None:
    img = cv2.imread(str(ROOT / "samples" / "bus.jpg"))
    pt = YOLO("yolo11n-pose.pt")
    onnx_path = pt.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    res = {"cpu": platform.processor() or platform.machine(), "threads": __import__("os").cpu_count(),
           "pytorch": bench(pt, img), "onnxruntime": bench(YOLO(onnx_path, task="pose"), img)}
    out = ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
