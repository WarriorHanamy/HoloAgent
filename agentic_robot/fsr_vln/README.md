# FSR-VLN
[![Project](https://img.shields.io/badge/📖-Project-blue)](https://horizonrobotics.github.io/robot_lab/fsr-vln)
[![📄 arXiv](https://img.shields.io/badge/📄-arXiv-b31b1b)](https://arxiv.org/abs/2509.13733)
[![🤗 Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/HorizonRobotics/fsrvln_datasets)
[![Docker](https://img.shields.io/badge/Docker-Image-2496ED?logo=docker&logoColor=white)](https://github.com/users/zhaoyu199201/packages/container/package/holoagent)
[![中文介绍](https://img.shields.io/badge/中文介绍-07C160?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/HqnBlTNqOL3Z4Kg8tLHCSw)


This repository is the official implementation of the paper:

> **FSR-VLN: Fast and Slow Reasoning for Vision-Language Navigation with Hierarchical Multi-modal Scene Graph**


<img src="../../docs/assets/FSR_VLN_framework.png" alt="Overall Framework" width="700"/>


## 📔 Introduction
***FSR-VLN*** is a core component of the HoloAgent framework. It provides natural language guided navigation and intelligent interaction for general-purpose robots, and is built on core agent components such as mapping and localization, multimodal perception, decision-making and planning, and memory management. At its core, FSR-VLN is a vision–language navigation system that integrates a Hierarchical Multi-modal Scene Graph (HMSG) for coarse-to-fine environment representation with Fast-to-Slow Navigation Reasoning (FSR), leveraging VLM-driven refinement to enable efficient, real-time, long-range spatial reasoning.


## 🏗 Setup
Following instructions are for an Ubuntu>=20.04 system with CUDA support (single GPU ≥ 12 GB VRAM is sufficient for offline reproduction; no ROS 2 or training required).

### Option A: `uv` Setup (Recommended)
```bash
cd agentic_robot/fsr_vln

# 1. Create Python 3.10 virtual environment and install dependencies
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  -r requirements-uv.txt

# 2. Install SAM 2 (SAM2_BUILD_CUDA=0 avoids requiring system nvcc;
#    --no-deps preserves torch==2.5.0+cu121)
SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python \
  --no-build-isolation --no-deps \
  "git+https://github.com/facebookresearch/sam2.git"

# 3. Install fsr_vln in editable mode
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .
```

### Option B: Conda Setup
```bash
# conda env
conda env create -f environment.yaml
conda activate holoagent_semantic_mapping
```

```bash
# sam2 install
cd /path/segment-anything-2
pip install -e . --no-build-isolation
# If the torch version has been changed after installing sam2, please reinstall torch and torchvision
# pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

```bash
# set up the hmsg python package
pip install -e .
```

### Download Pretrained Checkpoints
Set a data root outside the repository:
```bash
export HOLOAGENT_DATA_ROOT=/path/to/fsr_vln_data
mkdir -p "${HOLOAGENT_DATA_ROOT}/checkpoints/sam2"

# Download SAM 2.1 hiera_large checkpoint (used by G1 configs)
curl -L -o "${HOLOAGENT_DATA_ROOT}/checkpoints/sam2/sam2.1_hiera_large.pt" \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```
`SigLIP-384` (`hf-hub:timm/ViT-SO400M-14-SigLIP-384`) is downloaded automatically by `open_clip` on first run.


## 🖼️ Dataset Preparation
FSR-VLN uses the G1 RGB-D sequences from [HorizonRobotics/fsrvln_datasets](https://huggingface.co/datasets/HorizonRobotics/fsrvln_datasets) organized under `${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking/data`:

```bash
mkdir -p "${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking/data/input/datasets/G1/decoded_scans"
curl -L -o "${HOLOAGENT_DATA_ROOT}/icra_sh3f.zip" \
  https://huggingface.co/datasets/HorizonRobotics/fsrvln_datasets/resolve/main/icra_sh3f.zip
unzip -q "${HOLOAGENT_DATA_ROOT}/icra_sh3f.zip" \
  -d "${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking/data/input/datasets/G1/decoded_scans"
ln -sfn icra_sh3f \
  "${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking/data/input/datasets/G1/decoded_scans/sh_3f"
```

### Data directory layout
```text
${HOLOAGENT_DATA_ROOT}/
├── checkpoints/
│   └── sam2/
│       └── sam2.1_hiera_large.pt
└── VLN/instance_tracking/data/                     (--data_dir)
    ├── input/
    │   └── datasets/
    │       └── G1/                                 (--dataset_name)
    │           └── decoded_scans/
    │               ├── icra_sh3f/                  (unzipped from HF)
    │               ├── sh_3f -> icra_sh3f          (scene_name)
    │               ├── ic_3f/
    │               ├── ic_4f/
    │               └── ic_7f/
    └── output/
        └── G1/
            └── holoagent_mapping-g1/               (--experiment_name)
                └── sh_3f/
```

`run_holoagent_mapping.py` reads scene data from:
```text
${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking/data/input/datasets/G1/decoded_scans/<scene_name>
```

and writes mapping / HMSG results to:
```text
${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking/data/output/G1/holoagent_mapping-g1/<scene_name>
```

### Data flow
```text
G1 RGB-D sequences
        │
        ▼
instance mapping (SAM 2.1 + SigLIP-384)
        │
        ▼
OVO output (ovo_map.ckpt)
        │
        ▼
hierarchical multi-modal scene graphs (HMSG)
        │
        ▼
visualization / graph query / navigation reasoning
```


## :rocket: Run
### 1. Create hierarchical multi-modal scene graphs:
Use the maintained G1 entry script (defaults to `CUDA_VISIBLE_DEVICES=0` and `.venv/bin/python` if present):
```bash
export HOLOAGENT_DATA_ROOT=/path/to/fsr_vln_data
cd agentic_robot/fsr_vln/run_bash
bash run_g1.sh
```

`run_g1.sh` launches `run_holoagent_mapping.py` with:
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_holoagent_mapping.py \
 --data_dir "${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking/data" \
 --dataset_name G1 \
 --experiment_name holoagent_mapping-g1 \
 --scenes_list sh_3f \
 --build_hmsg \
 --run \
 --hmsg_config_path configs/hmsg_bridge_sh3f.yaml \
 --cfg_path configs/ovo_g1_sh3f.yaml
```

This stage runs instance mapping first and then builds the HMSG scene graph for each scene listed in `--scenes_list`. For the default script, the input scene is `sh_3f`, and the output will be saved under:
```text
${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking/data/output/G1/holoagent_mapping-g1/sh_3f/graph_<timestamp>/
```

### 2. Query & evaluate HMSG (Fast Matching / Native Antigravity Slow Reasoning)

FSR-VLN's LLM instruction parsing and VLM slow-reasoning verification natively run on **Antigravity Gemini 4 Neon** via the official local ACP protocol (`agy acp --stdio`) with native base64 image blocks (no reverse proxy, no external API key, and no Aliyun OSS upload needed).

- **Fast-to-Slow Reasoning with Antigravity Gemini 4 Neon (High — default)**:
  ```bash
  export HOLOAGENT_DATA_ROOT=/path/to/fsr_vln_data
  cd agentic_robot/fsr_vln
  .venv/bin/python visualize_query_graph/visualize_query_graph_icra_sh3f.py \
    main.fast_slow_method=slow_reasoning main.use_gpt=true \
    models.llm.backend=antigravity models.llm.effort=high
  ```
- **Switch to Antigravity Gemini 4 Neon (Extra / XHigh)** (via Hydra flag or env var):
  ```bash
  .venv/bin/python visualize_query_graph/visualize_query_graph_icra_sh3f.py \
    main.fast_slow_method=slow_reasoning main.use_gpt=true \
    models.llm.effort=extra
  # Equivalently:
  # ANTIGRAVITY_EFFORT=extra .venv/bin/python ...
  # ANTIGRAVITY_MODEL=gemini-4-neon-xhigh .venv/bin/python ...
  ```
- **Pure offline Fast Matching (no LLM/VLM calls at all)**:
  ```bash
  .venv/bin/python visualize_query_graph/visualize_query_graph_icra_sh3f.py \
    main.fast_slow_method=fast_match main.use_gpt=false models.llm.backend=offline
  ```
- *(Optional)* **OpenAI / Azure OpenAI fallback**:
  ```bash
  export FSR_VLN_LLM_BACKEND=openai OPENAI_API_KEY=your_key
  # Optional: export OPENAI_BASE_URL=... OPENAI_MODEL=gpt-4o
  # Or Azure: export FSR_VLN_LLM_BACKEND=azure AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_DEPLOYMENT=...
  ```
> Set `FSR_VLN_HEADLESS=1` when running without a display to skip Open3D window rendering while still saving `.ply` and `all_results.json`. Pass `+main.max_queries=N` for quick smoke tests.

### 3. Visualize OVO scene output
```bash
.venv/bin/python visualize_scene.py \
  --working_dir "${HOLOAGENT_DATA_ROOT}/VLN/instance_tracking" \
  --run_path data/output/G1/holoagent_mapping-g1/sh_3f \
  --visualize_obj
```

The visualization stage reads the generated graph files from the mapping output directory and renders the scene graph for inspection.