"""Helpers for running perception on exported AerialVLN/AirVLN frames.

The full AerialVLN setup uses AirSim/Unreal simulators. For this perception
package, the useful integration boundary is an exported RGB frame plus optional
simulator metadata such as pose, heading, altitude, timestamp, or instruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGE_DIR_NAMES = ("images", "rgb", "frames", "observations")


@dataclass
class AerialVLNFrame:
    """One exported AerialVLN frame and optional simulator metadata."""

    image_path: Path
    frame_id: str
    episode_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def discover_aerialvln_frames(
    root: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
    image_dir_names: Sequence[str] = DEFAULT_IMAGE_DIR_NAMES,
) -> List[AerialVLNFrame]:
    """Discover image frames under an AerialVLN export or generic image folder."""
    root_path = Path(root)
    image_paths = _discover_image_paths(root_path, image_dir_names=image_dir_names)
    metadata_by_key = load_frame_metadata(metadata_path) if metadata_path else {}

    frames: List[AerialVLNFrame] = []
    for image_path in image_paths:
        frame_id = image_path.stem
        metadata = _metadata_for_frame(metadata_by_key, image_path)
        episode_id = _episode_id_for_image(root_path, image_path, metadata)
        frames.append(
            AerialVLNFrame(
                image_path=image_path,
                frame_id=frame_id,
                episode_id=episode_id,
                metadata=metadata,
            )
        )

    if limit is not None:
        return frames[:limit]
    return frames


def load_frame_metadata(metadata_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load optional frame metadata from a JSON file.

    Supported shapes:
    - {"frame_001": {...}, "frame_002.jpg": {...}}
    - [{"image": "frame_001.jpg", ...}, {"image_path": "...", ...}]
    - {"frames": [{"image": "frame_001.jpg", ...}]}
    """
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as file:
        raw_metadata = json.load(file)

    if isinstance(raw_metadata, dict) and "frames" in raw_metadata:
        return _metadata_list_to_mapping(raw_metadata["frames"])

    if isinstance(raw_metadata, dict):
        return {str(key): _as_metadata_dict(value) for key, value in raw_metadata.items()}

    if isinstance(raw_metadata, list):
        return _metadata_list_to_mapping(raw_metadata)

    raise ValueError("Unsupported metadata JSON format.")


def _discover_image_paths(
    root_path: Path,
    image_dir_names: Sequence[str],
) -> List[Path]:
    if root_path.is_file() and root_path.suffix.lower() in IMAGE_SUFFIXES:
        return [root_path]

    if not root_path.is_dir():
        raise FileNotFoundError(f"AerialVLN path not found: {root_path}")

    candidate_dirs = [root_path]
    for dir_name in image_dir_names:
        candidate = root_path / dir_name
        if candidate.is_dir():
            candidate_dirs.insert(0, candidate)

    image_paths: List[Path] = []
    for directory in candidate_dirs:
        image_paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if image_paths:
            break

    return sorted(image_paths)


def _metadata_list_to_mapping(items: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
    metadata_by_key: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        image_key = (
            item.get("image")
            or item.get("image_path")
            or item.get("frame")
            or item.get("frame_id")
            or item.get("filename")
        )
        if image_key is None:
            continue

        metadata_by_key[str(image_key)] = dict(item)

    return metadata_by_key


def _metadata_for_frame(
    metadata_by_key: Dict[str, Dict[str, Any]],
    image_path: Path,
) -> Dict[str, Any]:
    keys = (
        str(image_path),
        image_path.name,
        image_path.stem,
        image_path.as_posix(),
    )
    for key in keys:
        if key in metadata_by_key:
            return dict(metadata_by_key[key])
    return {}


def _episode_id_for_image(
    root_path: Path,
    image_path: Path,
    metadata: Dict[str, Any],
) -> Optional[str]:
    explicit_episode = (
        metadata.get("episode_id")
        or metadata.get("trajectory_id")
        or metadata.get("path_id")
    )
    if explicit_episode is not None:
        return str(explicit_episode)

    try:
        relative_parent = image_path.relative_to(root_path).parent
    except ValueError:
        relative_parent = image_path.parent

    if str(relative_parent) in ("", "."):
        return None
    return relative_parent.as_posix()


def _as_metadata_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}
