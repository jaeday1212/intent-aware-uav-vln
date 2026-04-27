"""Utilities for converting detections into a structured scene dictionary."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DetectionDict = Dict[str, Any]
SceneDict = Dict[str, Any]


def build_scene_dict(
    detections: List[DetectionDict],
    image_width: int,
    image_height: int,
    source_path: Optional[str] = None,
) -> SceneDict:
    """Convert detector output into scene objects with derived attributes."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")

    objects: List[DetectionDict] = []
    for detection in detections:
        bbox = [float(value) for value in detection.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])]
        if len(bbox) != 4:
            bbox = [0.0, 0.0, 0.0, 0.0]

        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        center_x = x1 + (width / 2.0)
        center_y = y1 + (height / 2.0)

        objects.append(
            {
                "label": str(detection.get("label", "unknown")),
                "bbox_xyxy": bbox,
                "confidence": float(detection.get("confidence", 0.0)),
                "center": [center_x, center_y],
                "width": width,
                "height": height,
                "area": width * height,
                "region_horizontal": _horizontal_region(center_x, image_width),
                "region_vertical": _vertical_region(center_y, image_height),
            }
        )

    return {
        "image": {
            "width": int(image_width),
            "height": int(image_height),
            "source_path": source_path,
        },
        "objects": objects,
    }


def _horizontal_region(center_x: float, image_width: int) -> str:
    third = image_width / 3.0
    if center_x < third:
        return "left"
    if center_x < 2.0 * third:
        return "center"
    return "right"


def _vertical_region(center_y: float, image_height: int) -> str:
    third = image_height / 3.0
    if center_y < third:
        return "top"
    if center_y < 2.0 * third:
        return "middle"
    return "bottom"
