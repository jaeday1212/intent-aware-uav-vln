# Intent-Aware UAV VLN

This repository contains the shared graduate project work for the intent-aware UAV vision-language navigation pipeline.

## Current Contribution: Role 3 Perception

Role 3 perception code has been added under:

```text
role3_perception/
```

This module converts UAV/drone image frames into structured belief-state JSON for downstream intent generation and scoring.

Current perception flow:

```text
Image frame -> YOLO detector -> Semantic mapper -> BeliefState JSON
```

The imported work includes:

- `role3_perception/perception/`: reusable Python perception package
- `role3_perception/scripts/demo_perception.py`: run inference on generic images or image folders
- `role3_perception/scripts/demo_aerialvln_perception.py`: run inference on exported AerialVLN/AirVLN frames
- `role3_perception/scripts/export_aerialvln_headless_frame.py`: helper for exporting one RGB frame from a running AirVLN simulator
- `role3_perception/scripts/aerialvln_headless_server.sh`: helper for launching the AirVLN simulator server
- `role3_perception/docs/aerialvln_headless_setup.md`: notes for headless/offscreen AirVLN use
- `role3_perception/tests/`: unit tests for belief-state construction, semantic mapping, and adapters

## Quick Start

From the repository root:

```bash
cd role3_perception
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run perception on one image:

```bash
python scripts/demo_perception.py path/to/frame.jpg --json --save-vis
```

Run perception on an image directory:

```bash
python scripts/demo_perception.py path/to/images --limit 5 --json --save-vis
```

Outputs are written under:

```text
role3_perception/outputs/json/
role3_perception/outputs/annotated/
```

## AerialVLN / AirVLN Boundary

AerialVLN/AirVLN is simulator-backed. The perception code currently integrates at the RGB-frame boundary:

```text
AirVLN/AirSim RGB frame + optional simulator metadata -> perception pipeline -> BeliefState
```

For exported AerialVLN frames:

```bash
cd role3_perception
python scripts/demo_aerialvln_perception.py path/to/aerialvln_export/images --limit 5 --save-vis
```

For exported frames plus metadata:

```bash
python scripts/demo_aerialvln_perception.py path/to/aerialvln_export --metadata path/to/metadata.json --limit 5 --save-vis
```

Simulator/headless setup notes are in:

```text
role3_perception/docs/aerialvln_headless_setup.md
```

Important simulator distinction: use offscreen/headless rendering that still produces RGB camera images. Do not disable rendering entirely, because the detector needs image frames.

## Tests

Run Role 3 tests from inside the module directory:

```bash
cd role3_perception
pytest tests
```

## More Documentation

The detailed Role 3 README is:

```text
role3_perception/README.md
```
