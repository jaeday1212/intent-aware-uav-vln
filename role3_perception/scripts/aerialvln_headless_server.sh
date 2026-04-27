#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_AIRVLN_ROOT="$(cd "${SCRIPT_DIR}/../../AirVLN_ws/AirVLN" 2>/dev/null && pwd || true)"

AIRVLN_ROOT="${AIRVLN_ROOT:-${DEFAULT_AIRVLN_ROOT}}"
AIRVLN_GPUS="${AIRVLN_GPUS:-0}"
AIRVLN_PORT="${AIRVLN_PORT:-30000}"

if [[ -z "${AIRVLN_ROOT}" || ! -f "${AIRVLN_ROOT}/airsim_plugin/AirVLNSimulatorServerTool.py" ]]; then
  echo "Set AIRVLN_ROOT to the cloned AirVLN directory." >&2
  exit 1
fi

cd "${AIRVLN_ROOT}"
exec python -u ./airsim_plugin/AirVLNSimulatorServerTool.py \
  --gpus "${AIRVLN_GPUS}" \
  --port "${AIRVLN_PORT}" \
  "$@"
