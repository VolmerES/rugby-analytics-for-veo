from __future__ import annotations

import json
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2

from rugby_vision.video.frames import iter_frames
from rugby_vision.video.metadata import inspect_video
from rugby_vision.visualization.overlay import draw_observations


def select_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA. Usa --device cpu.")
        return "0"
    return "cpu"


def observations_from_result(result, frame_index: int, timestamp: float, tracking: bool) -> list[dict]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    coordinates = boxes.xyxy.cpu().tolist()
    scores = boxes.conf.cpu().tolist()
    ids = boxes.id.int().cpu().tolist() if tracking and boxes.id is not None else [None] * len(scores)
    records = []
    for box, score, track_id in zip(coordinates, scores, ids):
        x1, y1, x2, y2 = [round(float(value), 2) for value in box]
        records.append({
            "frame": frame_index,
            "timestamp": round(timestamp, 6),
            "track_id": track_id,
            "class": "person",
            "bbox": [x1, y1, x2, y2],
            "center_x": round((x1 + x2) / 2, 2),
            "center_y": round((y1 + y2) / 2, 2),
            "width": round(x2 - x1, 2),
            "height": round(y2 - y1, 2),
            "confidence": round(float(score), 5),
        })
    return records


def run_pipeline(video_path: str | Path, config: dict, *, output_dir: str | Path | None = None,
                 device: str = "auto", tracking: bool = True, max_frames: int | None = None) -> dict:
    from ultralytics import YOLO
    import torch
    import ultralytics

    metadata = inspect_video(video_path)
    fps = metadata["fps"]
    if fps is None or fps <= 0:
        raise ValueError("El vídeo no indica FPS. No se puede crear un overlay con tiempo fiable.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames debe ser positivo")
    output = Path(output_dir or config["output"]["directory"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "video_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    records_path = output / ("tracks.jsonl" if tracking else "detections.jsonl")
    overlay_path = output / ("tracking_overlay.mp4" if tracking else "detections_overlay.mp4")
    run_path = output / "analysis_run.json"

    selected = select_device(device)
    model_name = config["detection"]["model"]
    model = YOLO(model_name)
    fourcc = cv2.VideoWriter_fourcc(*config["output"]["codec"])
    writer = cv2.VideoWriter(str(overlay_path), fourcc, fps,
                             (metadata["width"], metadata["height"]))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el MP4 de salida: {overlay_path}")
    processed = 0
    observations = 0
    identified = 0
    started = time.monotonic()
    try:
        with records_path.open("w", encoding="utf-8") as records:
            for frame_index, pts, frame in iter_frames(video_path):
                if max_frames is not None and processed >= max_frames:
                    break
                timestamp = pts if pts is not None else frame_index / fps
                options = dict(conf=config["detection"]["confidence"],
                               imgsz=config["detection"]["image_size"],
                               classes=[0], device=selected, verbose=False)
                if tracking:
                    result = model.track(frame, persist=True, tracker=config["tracking"]["tracker"],
                                         **options)[0]
                else:
                    result = model.predict(frame, **options)[0]
                found = observations_from_result(result, frame_index, timestamp, tracking)
                for item in found:
                    records.write(json.dumps(item, ensure_ascii=False) + "\n")
                writer.write(draw_observations(frame, found, tracking))
                processed += 1
                observations += len(found)
                identified += sum(item["track_id"] is not None for item in found)
                if processed % 100 == 0:
                    elapsed = max(time.monotonic() - started, 0.001)
                    total = metadata["frames"]
                    progress = f"{100 * processed / total:.1f}%" if total else "?%"
                    print(f"{progress}  {processed} frames  {processed / elapsed:.1f} FPS procesados", flush=True)
    finally:
        writer.release()
    elapsed = time.monotonic() - started
    status = {
        "analysis_run_id": str(uuid.uuid4()),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": metadata["path"],
        "model": model_name,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "device": selected,
        "gpu": torch.cuda.get_device_name(0) if selected == "0" else None,
        "hardware": platform.platform(),
        "parameters": {"detection": config["detection"], "tracking": config["tracking"] if tracking else None},
        "source_fps": fps,
        "source_resolution": [metadata["width"], metadata["height"]],
        "processed_frames": processed,
        "person_observations": observations,
        "observations_with_track_id": identified,
        "processing_fps": round(processed / elapsed, 2) if elapsed else None,
        "elapsed_seconds": round(elapsed, 2),
        "partial": max_frames is not None and (metadata["frames"] is None or processed < metadata["frames"]),
        "overlay": str(overlay_path),
        "records": str(records_path),
    }
    run_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    return status
