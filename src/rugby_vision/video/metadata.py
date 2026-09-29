from __future__ import annotations

from pathlib import Path

import av


def inspect_video(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"No existe el vídeo: {source}")
    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise ValueError(f"El archivo no tiene una pista de vídeo: {source}")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate else None
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else float(container.duration / av.time_base)
                if container.duration is not None
                else None
            )
            rotation = stream.metadata.get("rotate") or container.metadata.get("rotate")
            return {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "duration_seconds": duration,
                "fps": fps,
                "frames": stream.frames or None,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "codec": stream.codec_context.name,
                "bitrate": stream.bit_rate or container.bit_rate or None,
                "orientation_degrees": int(rotation) if rotation else 0,
                "has_audio": bool(container.streams.audio),
                "video_start_seconds": (
                    float(stream.start_time * stream.time_base)
                    if stream.start_time is not None and stream.time_base is not None
                    else None
                ),
                "timestamp_source": "frame_pts_when_available",
            }
    except av.FFmpegError as exc:
        raise ValueError(f"No se pudo leer el vídeo {source}: {exc}") from exc
