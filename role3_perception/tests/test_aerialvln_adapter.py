import json

from PIL import Image

from perception.aerialvln_adapter import discover_aerialvln_frames, load_frame_metadata


def test_discover_aerialvln_frames_from_images_dir_with_metadata(tmp_path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "frame_001.jpg"
    Image.new("RGB", (32, 32), color="white").save(image_path)

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "frame_001.jpg": {
                    "altitude_m": 25.0,
                    "heading_deg": 90.0,
                    "episode_id": "episode_a",
                }
            }
        ),
        encoding="utf-8",
    )

    frames = discover_aerialvln_frames(tmp_path, metadata_path=metadata_path)

    assert len(frames) == 1
    assert frames[0].image_path == image_path
    assert frames[0].frame_id == "frame_001"
    assert frames[0].episode_id == "episode_a"
    assert frames[0].metadata["altitude_m"] == 25.0


def test_load_frame_metadata_from_frames_list(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "image": "frame_001.jpg",
                        "timestamp": "2026-04-21T12:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    metadata = load_frame_metadata(metadata_path)

    assert metadata["frame_001.jpg"]["timestamp"] == "2026-04-21T12:00:00Z"
