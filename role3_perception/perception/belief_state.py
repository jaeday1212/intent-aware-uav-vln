"""Belief state representation for downstream intent reasoning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BeliefState:
    """Structured world model derived from a single UAV image."""

    image_width: int
    image_height: int
    source_path: Optional[str] = None
    objects: List[Dict[str, Any]] = field(default_factory=list)
    drone_state: Optional[Dict[str, Any]] = None

    @classmethod
    def from_scene_dict(cls, scene_dict: Dict[str, Any]) -> "BeliefState":
        image = scene_dict.get("image", {})
        return cls(
            image_width=int(image.get("width", 0)),
            image_height=int(image.get("height", 0)),
            source_path=image.get("source_path"),
            objects=list(scene_dict.get("objects", [])),
            drone_state=scene_dict.get("drone_state"),
        )

    def get_objects(self) -> List[Dict[str, Any]]:
        return list(self.objects)

    def get_objects_by_label(self, label: str) -> List[Dict[str, Any]]:
        return [obj for obj in self.objects if obj.get("label") == label]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": {
                "width": self.image_width,
                "height": self.image_height,
                "source_path": self.source_path,
            },
            "objects": list(self.objects),
            "drone_state": self.drone_state,
        }

    def summary(self) -> str:
        if not self.objects:
            return (
                f"BeliefState(image={self.image_width}x{self.image_height}, "
                "objects=0)"
            )

        counts = Counter(obj.get("label", "unknown") for obj in self.objects)
        labels = ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))
        return (
            f"BeliefState(image={self.image_width}x{self.image_height}, "
            f"objects={len(self.objects)}, labels=[{labels}])"
        )
