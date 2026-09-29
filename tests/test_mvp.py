import json
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from rugby_vision.cli import main
from rugby_vision.config import load_config
from rugby_vision.pipeline import observations_from_result
from rugby_vision.video.metadata import inspect_video


def test_metadata_and_inspect_command(tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    result = inspect_video(video)
    assert result["width"] == 64
    assert result["height"] == 48
    assert result["frames"] == 3
    assert result["fps"] == 10
    assert result["has_audio"] is False
    assert main(["inspect-video", str(video), "--output-dir", str(tmp_path / "out")]) == 0
    saved = json.loads((tmp_path / "out" / "video_metadata.json").read_text())
    assert saved["path"] == str(video.resolve())


def test_config_validation(tmp_path: Path):
    config = load_config()
    assert config["tracking"]["tracker"] == "bytetrack.yaml"
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("detection: {confidence: 2}\ntracking: {}\noutput: {}\n")
    with pytest.raises(ValueError, match="confidence"):
        load_config(invalid)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def int(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes(SimpleNamespace):
    def __len__(self):
        return len(self.conf.values)


def test_tracking_record_preserves_id_and_bbox():
    boxes = FakeBoxes(xyxy=FakeTensor([[1, 2, 11, 22]]),
                      conf=FakeTensor([0.876]), id=FakeTensor([17]))
    result = SimpleNamespace(boxes=boxes)
    record = observations_from_result(result, 120, 4.8, True)[0]
    assert record["track_id"] == 17
    assert record["center_x"] == 6
    assert record["width"] == 10
    assert record["confidence"] == 0.876
