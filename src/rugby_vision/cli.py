from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

from rugby_vision.config import load_config
from rugby_vision.video.metadata import inspect_video


def system_info() -> dict:
    import cv2
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "opencv": cv2.__version__,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ffmpeg_on_path": shutil.which("ffmpeg"),
        "ffprobe_on_path": shutil.which("ffprobe"),
        "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / 1e9, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rugby", description="MVP de Rugby Vision Analyst")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("system-info", help="Muestra el entorno local")
    inspect = sub.add_parser("inspect-video", help="Guarda los metadatos del vídeo")
    inspect.add_argument("video", type=Path)
    inspect.add_argument("--output-dir", type=Path)
    for command in ("detect", "track", "process", "visualize"):
        current = sub.add_parser(command, help="Procesa el vídeo y genera artefactos visuales")
        current.add_argument("video", type=Path)
        current.add_argument("--config", type=Path)
        current.add_argument("--output-dir", type=Path)
        current.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        current.add_argument("--max-frames", type=int, help="Prueba rápida con los primeros N frames")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "system-info":
            result = system_info()
        elif args.command == "inspect-video":
            result = inspect_video(args.video)
            output = (args.output_dir or Path("data/processed")).resolve()
            output.mkdir(parents=True, exist_ok=True)
            (output / "video_metadata.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            from rugby_vision.pipeline import run_pipeline

            config = load_config(args.config)
            result = run_pipeline(args.video, config, output_dir=args.output_dir,
                                  device=args.device, tracking=args.command != "detect",
                                  max_frames=args.max_frames)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
