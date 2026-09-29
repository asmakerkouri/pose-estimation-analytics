"""python -m posekit.cli image.jpg [more.jpg ...] --out output/   (also works with a video file)"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .detector import PoseDetector
from .draw import draw_people
from .reps import RepCounter


def run_video(det: PoseDetector, path: str, out_dir: Path, joint: str = "left_knee") -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_dir / f"{Path(path).stem}_pose.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    counter, frames = RepCounter(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        people = det(frame)
        reps = counter.update(people[0].angles.get(joint) if people else None)
        out = draw_people(frame, people)
        cv2.putText(out, f"reps: {reps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        writer.write(out)
        frames += 1
    writer.release()
    return {"frames": frames, "reps": counter.reps}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", default="output")
    p.add_argument("--weights", default=None)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = PoseDetector(a.weights)
    for path in a.inputs:
        if Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            print(path, run_video(det, path, out))
            continue
        img = cv2.imread(path)
        people = det(img)
        cv2.imwrite(str(out / f"{Path(path).stem}_pose.jpg"), draw_people(img, people))
        (out / f"{Path(path).stem}_pose.json").write_text(json.dumps([p.to_dict() for p in people], indent=2))
        print(f"{path}: {len(people)} people -> {[p.posture.label for p in people]}")


if __name__ == "__main__":
    main()
