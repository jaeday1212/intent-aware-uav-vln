"""Object detector wrapper built around a pretrained YOLOv8 model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image

DetectionDict = Dict[str, Any]


class ObjectDetector:
    """Wrap a pretrained YOLOv8 detector and normalize its output."""

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        image_size: int = 1280,
        model: Optional[Any] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.model = model if model is not None else self._load_model(model_name_or_path)

    def _load_model(self, model_name_or_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import path depends on environment
            raise ImportError(
                "ultralytics is required for ObjectDetector. "
                "Install it with `pip install ultralytics`."
            ) from exc

        return YOLO(model_name_or_path)

    def detect(self, image: Union[str, Path, Image.Image, Any]) -> List[DetectionDict]:
        """Run inference and return a normalized list of detections."""
        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                verbose=False,
            )
        except RuntimeError as exc:
            message = str(exc)
            if "Numpy is not available" in message:
                raise RuntimeError(
                    "YOLO inference failed because PyTorch cannot use NumPy in the current "
                    "environment. This is usually caused by an incompatible torch/numpy "
                    "combination. Use a fresh environment and install a compatible stack, "
                    "for example: `python3.10 -m venv .venv && source .venv/bin/activate && "
                    "pip install 'numpy<2' pillow ultralytics pytest`."
                ) from exc
            raise

        normalized: List[DetectionDict] = []
        if not results:
            return normalized

        default_names = getattr(self.model, "names", {})
        for result in results:
            result_names = getattr(result, "names", default_names)
            boxes = getattr(result, "boxes", None)
            if not boxes:
                continue

            for box in boxes:
                label_index = int(self._extract_scalar(getattr(box, "cls", -1)))
                confidence = float(self._extract_scalar(getattr(box, "conf", 0.0)))
                bbox_xyxy = self._extract_bbox(getattr(box, "xyxy", []))

                normalized.append(
                    {
                        "label": self._resolve_label(label_index, result_names),
                        "bbox_xyxy": bbox_xyxy,
                        "confidence": confidence,
                    }
                )

        return normalized

    @staticmethod
    def _extract_scalar(value: Any) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        if isinstance(value, (list, tuple)):
            if not value:
                return 0.0
            return ObjectDetector._extract_scalar(value[0])
        return float(value)

    @staticmethod
    def _extract_bbox(value: Any) -> List[float]:
        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
            value = value[0]

        if not isinstance(value, Sequence) or len(value) != 4:
            return [0.0, 0.0, 0.0, 0.0]

        return [float(coord) for coord in value]

    @staticmethod
    def _resolve_label(label_index: int, names: Any) -> str:
        if isinstance(names, dict):
            return str(names.get(label_index, str(label_index)))
        if isinstance(names, list) and 0 <= label_index < len(names):
            return str(names[label_index])
        return str(label_index)
