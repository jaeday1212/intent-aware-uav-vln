# AerialVLN Headless Setup

This project uses AirVLN at the exported-frame boundary:

```text
AirVLN/AirSim headless scene -> RGB frame + metadata -> perception pipeline -> BeliefState JSON
```

The AirVLN simulator must still render camera images. Use offscreen rendering, not disabled rendering, because YOLO needs RGB frames.

## Local Workspace

The local AirVLN workspace for this machine is:

```text
/media/volume/safari_ty_volume/drone-project/AirVLN_ws
```

Expected layout:

```text
AirVLN_ws/
  AirVLN/
  DATA/data/aerialvln/
  DATA/data/aerialvln-s/
  DATA/models/ddppo-models/
  ENVs/
```

## Start The Headless Simulator Server

Use the AirVLN conda environment:

```bash
conda activate AirVLN
AIRVLN_GPUS=0 AIRVLN_PORT=30000 ./scripts/aerialvln_headless_server.sh
```

The upstream AirVLN server launcher writes AirSim settings with `ViewMode: NoDisplay` and launches Unreal scenes with:

```text
-RenderOffscreen -NoSound -NoVSync
```

Leave this server running while exporting frames.

## Export One AirVLN Frame

In another shell, still using the AirVLN environment:

```bash
conda activate AirVLN
python scripts/export_aerialvln_headless_frame.py \
  --airvln-root ../AirVLN_ws/AirVLN \
  --airvln-data-root ../AirVLN_ws/DATA/data \
  --dataset aerialvln \
  --split test \
  --episode-index 0 \
  --output-dir outputs/aerialvln_headless_export
```

This writes:

```text
outputs/aerialvln_headless_export/images/<episode>_start.png
outputs/aerialvln_headless_export/metadata.json
```

If colors look swapped in the saved frame, re-run the exporter with `--bgr-to-rgb`.

## Run Perception

Use the perception environment:

```bash
.venv/bin/python scripts/demo_aerialvln_perception.py \
  outputs/aerialvln_headless_export \
  --metadata outputs/aerialvln_headless_export/metadata.json \
  --save-vis
```

In this workspace, `.venv` is a Python 3.10 Conda prefix because the system Python is newer than the tested YOLO/NumPy stack. You can also activate it with `conda activate "$PWD/.venv"`.

Outputs are saved under:

```text
outputs/aerialvln/json/
outputs/aerialvln/annotated/
```

The AirVLN metadata is inserted into `drone_state`, including `episode_id`, `trajectory_id`, `scene_id`, instruction text, start pose, and simulator render mode.

## Notes

- The server needs the simulator environments under `../AirVLN_ws/ENVs/env_*/LinuxNoEditor/AirVLN.sh`.
- This host did not expose `nvidia-smi` during setup, so simulator launch could not be verified here. A GPU-capable runtime is expected for Unreal/AirSim offscreen rendering.
- Keep AirVLN dependencies separate from the perception environment. AirVLN pins older research packages; the perception environment only needs `numpy<2`, `pillow`, `ultralytics`, and `pytest`.
- The `AirVLN` conda environment is set up for simulator/frame export. The full AirVLN training/evaluation model stack and DD-PPO checkpoint are not required for this perception boundary.
