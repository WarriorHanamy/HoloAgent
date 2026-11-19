
#!/usr/bin/env bash
set -euo pipefail

# all_scenes=("sh_3f","ic_4f")
# all_scenes=("ic_7f","ic_3f")
SCENE_NAME="${SCENES_LIST:-sh_3f}"
all_scenes=("${SCENE_NAME}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSR_VLN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FSR_VLN_DIR}"

SCENE_TAG="${SCENE_NAME//_/}"
HMSG_CFG="${HMSG_CONFIG_PATH:-configs/hmsg_bridge_${SCENE_TAG}.yaml}"
OVO_CFG="${OVO_CONFIG_PATH:-configs/ovo_g1_${SCENE_TAG}.yaml}"
[[ -f "${HMSG_CFG}" ]] || HMSG_CFG="configs/hmsg_bridge_sh3f.yaml"
[[ -f "${OVO_CFG}" ]] || OVO_CFG="configs/ovo_g1_sh3f.yaml"

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
 --hmsg_config_path "${HMSG_CFG}" \
 --cfg_path "${OVO_CFG}"

# pkill -9 -f run_holoagent_mapping.py
