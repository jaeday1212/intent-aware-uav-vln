"""Run the perception pipeline on one or more sample images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.belief_state import BeliefState
from perception.detector_wrapper import ObjectDetector
from perception.pipeline import build_belief_state

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo the perception pipeline.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image paths or directories to process.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model name or path. Default: yolov8n.pt",
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
        "--json",
        action="store_true",
        help="Print the full belief state as JSON after the summary.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for saved outputs. Default: outputs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process after expanding directories.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save annotated images with bounding boxes.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Directory for annotated images. Default: <output-dir>/annotated",
    )
    return parser.parse_args()


def resolve_image_paths(inputs: List[str], limit: Optional[int] = None) -> List[Path]:
    image_paths: List[Path] = []

    for raw_input in inputs:
        path = Path(raw_input)
        if path.is_dir():
            directory_images = sorted(
                child for child in path.iterdir() if child.suffix.lower() in IMAGE_SUFFIXES
            )
            image_paths.extend(directory_images)
        elif path.is_file():
            image_paths.append(path)
        else:
            raise FileNotFoundError(f"Input not found: {path}")

    if limit is not None:
        return image_paths[:limit]
    return image_paths


def save_annotated_image(
    image_path: Path,
    belief_state: BeliefState,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        annotated = image.convert("RGB")

    draw = ImageDraw.Draw(annotated)
    line_width = max(2, min(6, belief_state.image_width // 300))

    for obj in belief_state.get_objects():
        bbox = obj.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(value) for value in bbox]
        label = str(obj.get("label", "unknown"))
        confidence = float(obj.get("confidence", 0.0))
        caption = f"{label} {confidence:.2f}"

        draw.rectangle((x1, y1, x2, y2), outline="red", width=line_width)
        text_bbox = draw.textbbox((x1, y1), caption)
        text_left, text_top, text_right, text_bottom = text_bbox
        text_height = text_bottom - text_top
        caption_top = max(0.0, y1 - text_height - 6.0)
        caption_box = (
            x1,
            caption_top,
            x1 + (text_right - text_left) + 8.0,
            caption_top + text_height + 6.0,
        )
        draw.rectangle(caption_box, fill="red")
        draw.text((x1 + 4.0, caption_top + 3.0), caption, fill="white")

    output_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
    annotated.save(output_path)
    return output_path


def save_belief_state_json(
    image_path: Path,
    belief_state: BeliefState,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}.json"
    output_path.write_text(json.dumps(belief_state.to_dict(), indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    image_paths = resolve_image_paths(args.inputs, limit=args.limit)
    if not image_paths:
        raise ValueError("No image files found in the provided inputs.")

    detector = ObjectDetector(
        model_name_or_path=args.model,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
    )
    output_root = Path(args.output_dir)
    json_dir = output_root / "json"
    vis_dir = Path(args.vis_dir) if args.vis_dir else output_root / "annotated"

    for image_path in image_paths:
        belief_state = build_belief_state(
            image=image_path,
            detector=detector,
        )
        json_path = save_belief_state_json(image_path, belief_state, json_dir)
        print(f"\nImage: {image_path}")
        print(belief_state.summary())
        print(f"Belief state JSON saved to: {json_path}")
        if args.json:
            print(json.dumps(belief_state.to_dict(), indent=2))
        if args.save_vis:
            output_path = save_annotated_image(image_path, belief_state, vis_dir)
            print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
