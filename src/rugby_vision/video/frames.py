from __future__ import annotations

from pathlib import Path
from typing import Iterator

import av


def iter_frames(path: str | Path) -> Iterator[tuple[int, float | None, object]]:
    """Produce un frame BGR cada vez, con tiempo relativo al primer PTS."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first_time = None
        for index, frame in enumerate(container.decode(stream)):
            absolute = float(frame.time) if frame.time is not None else None
            if absolute is not None and first_time is None:
                first_time = absolute
            timestamp = absolute - first_time if absolute is not None else None
            yield index, timestamp, frame.to_ndarray(format="bgr24")
