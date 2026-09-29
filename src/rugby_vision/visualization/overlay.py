from __future__ import annotations

import cv2


def draw_observations(frame, observations: list[dict], tracking: bool):
    for item in observations:
        x1, y1, x2, y2 = [int(value) for value in item["bbox"]]
        track_id = item["track_id"]
        color = (65, 210, 75) if track_id is not None else (0, 180, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = (f"TRACK {track_id}" if tracking and track_id is not None else "PERSON")
        label += f" {item['confidence']:.2f}"
        cv2.putText(frame, label, (x1, max(22, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return frame
