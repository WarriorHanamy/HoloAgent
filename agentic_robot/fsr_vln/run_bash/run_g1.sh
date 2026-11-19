
#!/usr/bin/env bash
set -euo pipefail

# all_scenes=("sh_3f","ic_4f")
# all_scenes=("ic_7f","ic_3f")
all_scenes=("${SCENES_LIST:-sh_3f}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSR_VLN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FSR_VLN_DIR}"

if [[ -z "${HOLOAGENT_DATA_ROOT:-}" ]]; then
  echo "Error: HOLOAGENT_DATA_ROOT is not set." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${FSR_VLN_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${FSR_VLN_DIR}/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" run_holoagent_mapping.py \
 --data_dir "${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking/data" \
 --dataset_name G1 \
 --experiment_name holoagent_mapping-g1 \
 --scenes_list "$(IFS=, ; echo "${all_scenes[*]}")" \
 --build_hmsg \
 --run \
 --hmsg_config_path configs/hmsg_bridge_sh3f.yaml \
 --cfg_path configs/ovo_g1_sh3f.yaml

# pkill -9 -f run_holoagent_mapping.py
