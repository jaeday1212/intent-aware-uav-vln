"""Run perception on exported AerialVLN/AirVLN RGB frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.aerialvln_adapter import AerialVLNFrame, discover_aerialvln_frames
from perception.detector_wrapper import ObjectDetector
from perception.pipeline import build_belief_state
from scripts.demo_perception import save_annotated_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the perception pipeline on exported AerialVLN frames."
    )
    parser.add_argument(
        "root",
        help="AerialVLN export root, image directory, or single image file.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON metadata file keyed by frame/image name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model name or path. Default: yolov8n.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold. Default: 0.25",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size passed to YOLO. Default: 1280",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/aerialvln",
        help="Root directory for saved outputs. Default: outputs/aerialvln",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save annotated frame images with bounding boxes.",
    )
    return parser.parse_args()


def output_stem(frame: AerialVLNFrame) -> str:
    if frame.episode_id:
        safe_episode = frame.episode_id.replace("/", "_")
        return f"{safe_episode}_{frame.frame_id}"
    return frame.frame_id


def save_frame_json(
    frame: AerialVLNFrame,
    belief_dict: dict,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem(frame)}.json"
    output_path.write_text(json.dumps(belief_dict, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    frames = discover_aerialvln_frames(
        root=args.root,
        metadata_path=args.metadata,
        limit=args.limit,
    )
    if not frames:
        raise ValueError(f"No RGB image frames found under {args.root}")

    detector = ObjectDetector(
        model_name_or_path=args.model,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
    )

    output_root = Path(args.output_dir)
    json_dir = output_root / "json"
    vis_dir = output_root / "annotated"

    for frame in frames:
        belief_state = build_belief_state(
            image=frame.image_path,
            detector=detector,
            drone_state=frame.metadata or None,
        )
        belief_dict = belief_state.to_dict()
        belief_dict["aerialvln"] = {
            "frame_id": frame.frame_id,
            "episode_id": frame.episode_id,
        }

        json_path = save_frame_json(frame, belief_dict, json_dir)
        print(f"\nFrame: {frame.image_path}")
        print(belief_state.summary())
        print(f"Belief state JSON saved to: {json_path}")

        if args.save_vis:
            vis_path = save_annotated_image(frame.image_path, belief_state, vis_dir)
            print(f"Annotated frame saved to: {vis_path}")


if __name__ == "__main__":
    main()
