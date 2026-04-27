"""High-level pipeline that builds a belief state from an image."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in the demo environment
    np = None

from PIL import Image

from perception.belief_state import BeliefState
from perception.detector_wrapper import ObjectDetector
from perception.semantic_mapper import build_scene_dict


def build_belief_state(
    image: Union[str, Path, Image.Image, Any],
    detector: Optional[ObjectDetector] = None,
    model_name_or_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
    image_size: int = 1280,
    drone_state: Optional[Dict[str, Any]] = None,
) -> BeliefState:
    """Run detector and semantic mapping to produce a belief state."""
    active_detector = (
        detector
        if detector is not None
        else ObjectDetector(
            model_name_or_path=model_name_or_path,
            confidence_threshold=confidence_threshold,
            image_size=image_size,
        )
    )
    width, height, source_path = _extract_image_metadata(image)
    detections = active_detector.detect(image)
    scene_dict = build_scene_dict(
        detections=detections,
        image_width=width,
        image_height=height,
        source_path=source_path,
    )
    scene_dict["drone_state"] = drone_state
    return BeliefState.from_scene_dict(scene_dict)


def _extract_image_metadata(
    image: Union[str, Path, Image.Image, Any],
) -> Tuple[int, int, Optional[str]]:
    if isinstance(image, (str, Path)):
        path = Path(image)
        with Image.open(path) as pil_image:
            width, height = pil_image.size
        return width, height, str(path)

    if isinstance(image, Image.Image):
        width, height = image.size
        return width, height, None

    if np is not None and isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        return int(width), int(height), None

    raise TypeError(
        "Unsupported image input. Expected a path, PIL image, or numpy array."
    )
