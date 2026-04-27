"""Export one RGB frame from a headless AirVLN simulator scene.

This script keeps the simulator boundary separate from perception inference:
it writes an image plus metadata JSON that `demo_aerialvln_perception.py` can
consume in the normal perception environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AIRVLN_WORKSPACE = PROJECT_ROOT.parent / "AirVLN_ws"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a headless AirVLN RGB frame for the perception pipeline."
    )
    parser.add_argument(
        "--airvln-root",
        default=str(DEFAULT_AIRVLN_WORKSPACE / "AirVLN"),
        help="Path to the cloned AirVLN repository.",
    )
    parser.add_argument(
        "--airvln-data-root",
        default=str(DEFAULT_AIRVLN_WORKSPACE / "DATA" / "data"),
        help="Path containing AirVLN dataset folders.",
    )
    parser.add_argument(
        "--dataset",
        choices=("aerialvln", "aerialvln-s"),
        default="aerialvln",
        help="AirVLN annotation dataset to read.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Annotation split JSON name without extension. Default: test.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode index in the selected split when --episode-id is not set.",
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Specific AirVLN episode_id to capture.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="AirVLN simulator server host. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="AirVLN simulator server port. Default: 30000.",
    )
    parser.add_argument(
        "--airsim-timeout",
        type=int,
        default=120,
        help="AirSim client timeout in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/aerialvln_headless_export",
        help="Directory where images/ and metadata.json are written.",
    )
    parser.add_argument(
        "--image-name",
        default=None,
        help="Optional output image filename. Defaults to episode_id_start.png.",
    )
    parser.add_argument(
        "--bgr-to-rgb",
        action="store_true",
        help="Swap channel order before saving if the AirSim scene returns BGR.",
    )
    parser.add_argument(
        "--keep-scenes-open",
        action="store_true",
        help="Do not ask AirVLN to close scenes after capture.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait after setting the simulator pose before capture.",
    )
    parser.add_argument(
        "--capture-attempts",
        type=int,
        default=5,
        help="Maximum RGB capture attempts before failing validation.",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=1.0,
        help="Seconds to wait between capture attempts.",
    )
    parser.add_argument(
        "--min-mean",
        type=float,
        default=15.0,
        help="Minimum mean pixel intensity for a usable RGB frame.",
    )
    parser.add_argument(
        "--min-nonblack-ratio",
        type=float,
        default=0.05,
        help="Minimum fraction of pixels with any channel above 10.",
    )
    parser.add_argument(
        "--disable-frame-validation",
        action="store_true",
        help="Save the first captured frame even if it appears blank.",
    )
    return parser.parse_args()


def load_episode(
    data_root: Path,
    dataset: str,
    split: str,
    episode_index: int,
    episode_id: str | None,
) -> Dict[str, Any]:
    split_path = data_root / dataset / f"{split}.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"AirVLN split not found: {split_path}")

    raw = json.loads(split_path.read_text(encoding="utf-8"))
    episodes = raw.get("episodes") if isinstance(raw, dict) else raw
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"No episodes found in {split_path}")

    if episode_id:
        for episode in episodes:
            if str(episode.get("episode_id")) == episode_id:
                return episode
        raise ValueError(f"Episode id not found in {split_path}: {episode_id}")

    try:
        return episodes[episode_index]
    except IndexError as exc:
        raise IndexError(
            f"Episode index {episode_index} is outside split with {len(episodes)} episodes."
        ) from exc


def import_airvln_client(
    airvln_root: Path,
    project_prefix: Path,
    simulator_port: int,
) -> Tuple[Any, Any]:
    if not (airvln_root / "airsim_plugin" / "AirVLNSimulatorClientTool.py").is_file():
        raise FileNotFoundError(f"AirVLN client not found under: {airvln_root}")

    original_argv = sys.argv[:]
    original_path = sys.path[:]
    sys.path.insert(0, str(airvln_root))
    sys.argv = [
        original_argv[0],
        "--project_prefix",
        str(project_prefix),
        "--run_type",
        "eval",
        "--batchSize",
        "1",
        "--simulator_tool_port",
        str(simulator_port),
    ]

    try:
        import airsim
        from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
    finally:
        sys.argv = original_argv
        sys.path = original_path

    return AirVLNSimulatorClientTool, airsim


def episode_pose(episode: Dict[str, Any], airsim: Any) -> Any:
    position = episode["start_position"]
    rotation = episode["start_rotation"]
    return airsim.Pose(
        position_val=airsim.Vector3r(
            x_val=position[0],
            y_val=position[1],
            z_val=position[2],
        ),
        orientation_val=airsim.Quaternionr(
            x_val=rotation[1],
            y_val=rotation[2],
            z_val=rotation[3],
            w_val=rotation[0],
        ),
    )


def capture_rgb(
    client_tool: Any,
    episode: Dict[str, Any],
    airsim: Any,
    warmup_seconds: float,
    capture_attempts: int,
    capture_interval: float,
    min_mean: float,
    min_nonblack_ratio: float,
    validate_frame: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    pose = episode_pose(episode, airsim)
    if not client_tool.setPoses([[pose]]):
        raise RuntimeError("AirVLN failed to set the episode start pose.")

    if warmup_seconds > 0:
        time.sleep(warmup_seconds)

    last_rgb = None
    last_quality: Dict[str, float] = {}
    for attempt in range(1, max(capture_attempts, 1) + 1):
        responses = client_tool.getImageResponses(get_rgb=True, get_depth=False)
        if not responses or not responses[0] or not responses[0][0]:
            raise RuntimeError("AirVLN returned no RGB image response.")

        rgb = responses[0][0][0]
        if rgb is None:
            raise RuntimeError("AirVLN returned an empty RGB frame.")

        last_rgb = np.asarray(rgb, dtype=np.uint8)
        last_quality = frame_quality(last_rgb)
        print(
            "Capture attempt "
            f"{attempt}: mean={last_quality['mean']:.2f}, "
            f"std={last_quality['std']:.2f}, "
            f"nonblack={last_quality['nonblack_ratio']:.4f}"
        )
        if not validate_frame or frame_is_usable(
            last_quality,
            min_mean=min_mean,
            min_nonblack_ratio=min_nonblack_ratio,
        ):
            return last_rgb, last_quality

        if attempt < capture_attempts and capture_interval > 0:
            time.sleep(capture_interval)

    raise RuntimeError(
        "AirVLN captured frames, but they look blank or edge-only. "
        f"Last quality: {last_quality}. "
        "This usually means the simulator renderer is not producing shaded RGB "
        "in the current VM/runtime."
    )


def frame_quality(rgb: np.ndarray) -> Dict[str, float]:
    if rgb.size == 0:
        return {"mean": 0.0, "std": 0.0, "nonblack_ratio": 0.0}

    nonblack_ratio = float((rgb.max(axis=2) > 10).mean())
    return {
        "mean": float(rgb.mean()),
        "std": float(rgb.std()),
        "nonblack_ratio": nonblack_ratio,
    }


def frame_is_usable(
    quality: Dict[str, float],
    min_mean: float,
    min_nonblack_ratio: float,
) -> bool:
    return (
        quality["mean"] >= min_mean
        and quality["nonblack_ratio"] >= min_nonblack_ratio
    )


def metadata_for_frame(
    image_name: str,
    episode: Dict[str, Any],
    dataset: str,
    split: str,
    host: str,
    port: int,
) -> Dict[str, Any]:
    instruction = episode.get("instruction", {})
    if isinstance(instruction, dict):
        instruction_text = instruction.get("instruction_text")
    else:
        instruction_text = instruction

    return {
        "image": image_name,
        "episode_id": episode.get("episode_id"),
        "trajectory_id": episode.get("trajectory_id"),
        "scene_id": episode.get("scene_id"),
        "instruction": instruction_text,
        "start_position": episode.get("start_position"),
        "start_rotation": episode.get("start_rotation"),
        "source_dataset": dataset,
        "source_split": split,
        "simulator": {
            "host": host,
            "port": port,
            "rendering": "RenderOffscreen",
            "view_mode": "NoDisplay",
        },
    }


def main() -> None:
    args = parse_args()
    airvln_root = Path(args.airvln_root).expanduser().resolve()
    airvln_workspace = airvln_root.parent
    data_root = Path(args.airvln_data_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    episode = load_episode(
        data_root=data_root,
        dataset=args.dataset,
        split=args.split,
        episode_index=args.episode_index,
        episode_id=args.episode_id,
    )
    scene_id = episode.get("scene_id")
    if scene_id is None:
        raise ValueError("Selected episode does not contain a scene_id.")

    AirVLNSimulatorClientTool, airsim = import_airvln_client(
        airvln_root=airvln_root,
        project_prefix=airvln_workspace,
        simulator_port=args.port,
    )

    machines_info = [
        {
            "MACHINE_IP": args.host,
            "SOCKET_PORT": args.port,
            "MAX_SCENE_NUM": 1,
            "open_scenes": [scene_id],
        }
    ]

    client_tool = AirVLNSimulatorClientTool(machines_info=machines_info)
    try:
        client_tool.run_call(airsim_timeout=args.airsim_timeout)
        rgb, quality = capture_rgb(
            client_tool=client_tool,
            episode=episode,
            airsim=airsim,
            warmup_seconds=args.warmup_seconds,
            capture_attempts=args.capture_attempts,
            capture_interval=args.capture_interval,
            min_mean=args.min_mean,
            min_nonblack_ratio=args.min_nonblack_ratio,
            validate_frame=not args.disable_frame_validation,
        )
    finally:
        if not args.keep_scenes_open:
            try:
                client_tool.closeScenes()
            except Exception:
                pass

    if args.bgr_to_rgb:
        rgb = rgb[..., ::-1]

    image_name = args.image_name or f"{episode['episode_id']}_start.png"
    image_path = images_dir / image_name
    Image.fromarray(rgb, mode="RGB").save(image_path)

    metadata = metadata_for_frame(
        image_name=image_name,
        episode=episode,
        dataset=args.dataset,
        split=args.split,
        host=args.host,
        port=args.port,
    )
    metadata["frame_quality"] = quality
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps({"frames": [metadata]}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved RGB frame: {image_path}")
    print(f"Saved metadata: {metadata_path}")
    print()
    print("Run perception with:")
    print(
        ".venv/bin/python scripts/demo_aerialvln_perception.py "
        f"{output_dir} --metadata {metadata_path} --save-vis"
    )


if __name__ == "__main__":
    main()
