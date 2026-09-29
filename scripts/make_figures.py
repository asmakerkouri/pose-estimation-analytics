"""Regenerate README figures: python scripts/make_figures.py

- docs/images/gallery.jpg      skeletons + postures on sample images (Ultralytics samples + COCO)
- docs/images/rep_counter.png  rep counting on a simulated knee-angle signal
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from posekit.detector import PoseDetector  # noqa: E402
from posekit.draw import draw_people  # noqa: E402
from posekit.reps import RepCounter  # noqa: E402

OUT = ROOT / "docs" / "images"
COCO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_IDS = ["000000000625", "000000000510", "000000000542"]


def coco_images() -> list[Path]:
    data = ROOT / "data"
    folder = data / "coco128" / "images" / "train2017"
    if not folder.exists():
        data.mkdir(exist_ok=True)
        zpath = data / "coco128.zip"
        urllib.request.urlretrieve(COCO_URL, zpath)
        zipfile.ZipFile(zpath).extractall(data)
    return [folder / f"{i}.jpg" for i in COCO_IDS]


def tile(img: np.ndarray, h: int = 420) -> np.ndarray:
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h), interpolation=cv2.INTER_AREA)


def gallery(det: PoseDetector) -> None:
    paths = [ROOT / "samples" / "bus.jpg", *coco_images(), ROOT / "samples" / "zidane.jpg"]
    tiles = []
    for p in paths:
        img = cv2.imread(str(p))
        people = [x for x in det(img) if x.score > 0.5]
        tiles.append(tile(draw_people(img, people)))
    row1 = np.hstack(tiles[:3])
    row2 = np.hstack(tiles[3:])
    w = max(row1.shape[1], row2.shape[1])
    pad = lambda r: cv2.copyMakeBorder(r, 0, 6, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(252, 252, 251))  # noqa: E731
    grid = np.vstack([pad(row1), pad(row2)])
    grid = cv2.resize(grid, (1600, int(grid.shape[0] * 1600 / grid.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "gallery.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def rep_figure() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 1 / 15)                      # 20 s at 15 fps
    knee = 175 - 42 * (1 - np.cos(2 * np.pi * t / 4)) + rng.normal(0, 4, t.size)  # squat every 4 s
    c, reps = RepCounter(), []
    for a in knee:
        reps.append(c.update(float(a)))
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=130, facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(t, knee, color="#2a78d6", lw=2, label="knee angle (simulated, with keypoint noise)")
    ax.axhline(c.down_below, color="#52514e", lw=1, ls="--")
    ax.axhline(c.up_above, color="#52514e", lw=1, ls="--")
    ax.text(0.1, c.down_below - 8, "down threshold", color="#52514e", fontsize=9)
    ax.text(0.1, c.up_above + 3, "up threshold", color="#52514e", fontsize=9)
    for i in np.flatnonzero(np.diff(reps)) + 1:
        ax.annotate(f"rep {reps[i]}", (t[i], knee[i]), xytext=(t[i], 190), ha="center", fontsize=9,
                    color="#0b0b0b", arrowprops={"arrowstyle": "-", "color": "#b0afa8"})
    ax.set_ylim(70, 200)
    ax.set_xlabel("seconds", color="#52514e")
    ax.set_ylabel("angle (deg)", color="#52514e")
    ax.set_title(f"Squat counter: {c.reps} reps detected from the knee angle", loc="left", color="#0b0b0b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "rep_counter.png", facecolor="#fcfcfb")


def benchmark_figure() -> None:
    """Bar chart from docs/images/benchmark.json (produced by scripts/benchmark.py)."""
    import json

    b = json.loads((OUT / "benchmark.json").read_text())
    names = ["PyTorch", "ONNX Runtime"]
    med = [b["pytorch"]["median_ms"], b["onnxruntime"]["median_ms"]]
    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=130, facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    bars = ax.barh(names, med, color="#2a78d6", height=0.5)
    for bar, v in zip(bars, med, strict=True):
        ax.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.0f} ms", va="center", color="#0b0b0b")
    gain = round(100 * (1 - med[1] / med[0]))
    ax.set_title(f"YOLO11n-pose, 640 px, CPU ({b['threads']} threads): ONNX {gain} % faster",
                 loc="left", color="#0b0b0b", fontsize=11)
    ax.set_xlabel("median latency per image (ms)", color="#52514e")
    ax.set_xlim(0, max(med) * 1.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark.png", facecolor="#fcfcfb")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gallery(PoseDetector())
    rep_figure()
    benchmark_figure()
