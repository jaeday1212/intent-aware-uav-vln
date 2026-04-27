# Intent-Aware Drone Control Architecture

This repository currently contains the Role 3 perception work for the graduate project. The goal of this component is to turn UAV imagery into a structured, machine-readable belief state that downstream intent generation and scoring modules can consume.

Current pipeline:

```text
Image frame -> YOLO detector -> Semantic mapper -> BeliefState JSON
```

## What This Work Does

The current implementation provides:

- A YOLO-based object detector wrapper in `perception/detector_wrapper.py`.
- A semantic mapper in `perception/semantic_mapper.py` that enriches detections with center, width, height, area, and rough image region.
- A `BeliefState` data model in `perception/belief_state.py`.
- A pipeline entrypoint in `perception/pipeline.py`.
- A generic image demo script in `scripts/demo_perception.py`.
- An AerialVLN exported-frame adapter in `perception/aerialvln_adapter.py`.
- An AerialVLN-oriented demo script in `scripts/demo_aerialvln_perception.py`.
- JSON output saving for every processed image.
- Optional annotated images with bounding boxes.

The belief state stores:

- image width and height
- source image path
- detected objects
- each object label, bounding box, confidence, center, dimensions, area, and coarse region
- optional `drone_state` metadata for simulator pose/telemetry

## What This Work Does Not Do Yet

This repo does not currently:

- train or fine-tune a detector
- run the full AerialVLN/AirVLN simulator
- control a drone
- perform path planning or reinforcement learning
- parse full VLN trajectory/instruction labels
- maintain temporal memory across frames

For Phase 1, the focus is perception data flow: image in, structured belief state out.

## Install Dependencies

Use a fresh environment instead of the global/base Conda environment. This avoids common `torch`/`numpy` compatibility issues.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On this machine, system Python is newer than the tested perception stack, so `.venv` has been created as a Python 3.10 Conda prefix. Use either:

```bash
conda activate "$PWD/.venv"
```

or call its interpreter directly:

```bash
.venv/bin/python scripts/demo_perception.py path/to/frame.jpg --json --save-vis
```

`ultralytics` is required for real YOLO inference. The tests use mocked detector outputs and do not require downloading model weights.

On first real inference run, Ultralytics may download the requested model weights, such as `yolov8n.pt`, `yolov8x.pt`, or another supported checkpoint.

## Generic Image Demo

Run inference on one image:

```bash
python scripts/demo_perception.py path/to/frame.jpg --json --save-vis
```

Run inference on a directory, limited to five images:

```bash
python scripts/demo_perception.py VisDrone2019-DET-val/images --limit 5 --json --save-vis
```

Run a stronger/larger model with lower confidence for small aerial objects:

```bash
python scripts/demo_perception.py VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg --model yolov8x.pt --imgsz 1280 --conf 0.10 --json --save-vis
```

The demo defaults to `--imgsz 1280` to improve small-object recall in aerial imagery. Larger values may detect more small objects but will run slower.

## Outputs

Every processed image saves a belief-state JSON file by default:

```text
outputs/json/<image_stem>.json
```

If `--save-vis` is enabled, annotated images are saved here:

```text
outputs/annotated/<image_stem>_annotated.<ext>
```

Use `--output-dir` to change the output root:

```bash
python scripts/demo_perception.py path/to/frame.jpg --output-dir outputs/my_run --save-vis
```

## Belief State Example

Example object entry:

```json
{
  "label": "car",
  "bbox_xyxy": [100.0, 50.0, 220.0, 180.0],
  "confidence": 0.92,
  "center": [160.0, 115.0],
  "width": 120.0,
  "height": 130.0,
  "area": 15600.0,
  "region_horizontal": "center",
  "region_vertical": "middle"
}
```

Programmatic usage:

```python
from perception.pipeline import build_belief_state

belief = build_belief_state("path/to/frame.jpg")
print(belief.summary())
print(belief.to_dict())
```

## AerialVLN Integration

AerialVLN/AirVLN is simulator-backed. It uses AirSim/Unreal simulator environments plus VLN annotation data. The full simulator setup is large, so this repo first supports the useful perception boundary:

```text
exported AerialVLN RGB frame -> detector -> semantic mapper -> BeliefState
```

For this machine, the official AirVLN repository and datasets are set up as a sibling workspace:

```text
../AirVLN_ws/AirVLN
../AirVLN_ws/DATA/data/aerialvln
../AirVLN_ws/DATA/data/aerialvln-s
../AirVLN_ws/ENVs
```

Headless/offscreen use is documented in `docs/aerialvln_headless_setup.md`. The short version is:

```bash
conda activate AirVLN
AIRVLN_GPUS=0 ./scripts/aerialvln_headless_server.sh
```

Then export one simulator frame and metadata:

```bash
conda activate AirVLN
python scripts/export_aerialvln_headless_frame.py --dataset aerialvln --split test --episode-index 0
```

Finally run this perception pipeline on the exported frame:

```bash
.venv/bin/python scripts/demo_aerialvln_perception.py outputs/aerialvln_headless_export --metadata outputs/aerialvln_headless_export/metadata.json --save-vis
```

Run perception on a folder of exported AerialVLN frames:

```bash
python scripts/demo_aerialvln_perception.py path/to/aerialvln_export/images --limit 5 --save-vis
```

Attach optional simulator metadata from a JSON file:

```bash
python scripts/demo_aerialvln_perception.py path/to/aerialvln_export --metadata path/to/frame_metadata.json --limit 5
```

Example metadata shape:

```json
{
  "frame_001.jpg": {
    "timestamp": "2026-04-21T12:00:00Z",
    "altitude_m": 42.0,
    "heading_deg": 135.0,
    "gps": {"lat": 40.0, "lon": -74.0},
    "episode_id": "episode_001"
  }
}
```

This metadata is inserted into the final belief state as `drone_state`.

## Full Simulator Boundary

This repository does not currently launch or control the AirVLN simulator directly. For full online simulation, the simulator client should:

1. launch the AirVLN/AirSim environment
2. capture an RGB camera observation
3. read simulator pose/telemetry
4. call `build_belief_state(image, drone_state=telemetry)`
5. pass the resulting belief state to downstream intent modules

For efficient simulator execution, the group may run Unreal/AirSim in headless/offscreen mode. For perception, use an offscreen-rendering mode that still produces camera images. Do not disable rendering entirely, because YOLO needs RGB frames.

## Tests

Run tests with:

```bash
pytest tests
```

If `pytest` is missing:

```bash
pip install pytest
pytest tests
```
