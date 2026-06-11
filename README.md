# evalscope-custom

Evaluation pipeline for Vision-Language Models (VLMs) using [VLMEvalKit](https://github.com/modelscope/evalscope) with Thai and custom datasets.

This repository includes custom Thai benchmark datasets and scripts to reproduce evaluation results from our paper.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Serve the model (see Step 0 below)

# 3. Download and format Thai/custom datasets
python download_datasets.py
python format_eval.py

# 4. Run evaluation
python run_eval.py           # standard benchmarks
python run_eval_custom.py    # Thai / custom benchmarks

# 5. View results
python summary_report.py     # latest run
python summary_report.py --all  # all runs
```

---

## Steps

### 0. Serve the model

Start the model server before running evaluation. Default model: **[CMKL/MANGO-Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/CMKL/MANGO-Qwen3-Omni-30B-A3B-Instruct)**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model CMKL/MANGO-Qwen3-Omni-30B-A3B-Instruct \
  --served-model-name qwen3-omni \
  --port 8801 \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 4
```

To use a different model, update `--served-model-name` and the `type` field in the config files under `configs/`.

---

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Download datasets

```bash
python download_datasets.py
```

Downloads Thai/custom datasets from HuggingFace and saves them as `.tsv` files into `LMUData/`.
To add or remove datasets, edit the `DATASETS` list at the top of `download_datasets.py`.

> Standard benchmarks (`MMStar`, `SEEDBench_IMG`, etc.) are downloaded automatically by VLMEvalKit during evaluation.

---

### 3. Format datasets

```bash
python format_eval.py                              # all datasets
python format_eval.py MMStar_TH ScienceQA_TEST_TH  # specific datasets only
```

Converts raw TSV files from `LMUData/` into evalscope-compatible format and saves them to `LMUData_formatted/`.
Normalizes columns to `index | question | A–I | answer | image` and filters to multiple-choice questions only.

Some datasets (`MathVista`, `MathVerse`, `TGAT`) are loaded directly from HuggingFace during this step because their images require PIL decoding.

---

### 4. Run evaluation

**Standard benchmarks** (auto-downloaded by VLMEvalKit):
```bash
python run_eval.py
```
Uses `configs/standard.yaml`. Edit the `data` list in that file to select which benchmarks to run.

**Thai / custom benchmarks** (from `LMUData_formatted/`):
```bash
python run_eval_custom.py
```
Uses `configs/custom.yaml`. This script copies `LMUData_formatted/` → `~/LMUData` so VLMEvalKit can locate the files, then starts evaluation.

Results are saved to `outputs/` and a summary CSV is auto-generated after each run.

---

### 5. View results

Results are saved under `outputs/<timestamp>/`. A per-dataset accuracy CSV is auto-generated at the end of each run.

To regenerate the summary manually:
```bash
python summary_report.py          # latest run only
python summary_report.py --all    # all runs in outputs/
```

---

## Datasets

**Standard benchmarks** (auto-downloaded by VLMEvalKit):

| Dataset | Description |
|---------|-------------|
| `MMStar` | Multi-modal benchmark |
| `SEEDBench_IMG` | Image-based seed benchmark |
| `MMBench_DEV_EN` | MMBench English dev set |
| `ScienceQA_TEST` | Science QA test set |

**Thai / custom benchmarks** (downloaded and formatted manually):

| Dataset | Description | Dataset Card |
|---------|-------------|--------------|
| `MMStar_TH` | MMStar translated to Thai | [CMKL/MMStar_TH](https://huggingface.co/datasets/CMKL/MMStar_TH) |
| `SEEDBench_IMG_TH` | SEEDBench translated to Thai | [CMKL/SEEDBench_IMG_TH](https://huggingface.co/datasets/CMKL/SEEDBench_IMG_TH) |
| `MMBench_DEV_TH` | MMBench translated to Thai | [CMKL/MMBench_DEV_TH](https://huggingface.co/datasets/CMKL/MMBench_DEV_TH) |
| `MMBench95CI_DEV_TH` | MMBench 95% CI subset, Thai | [CMKL/MMBench95CI_DEV_TH](https://huggingface.co/datasets/CMKL/MMBench95CI_DEV_TH) — see [dataset README](docs/MMBench95CI_README.md) |
| `ScienceQA_TEST_TH` | ScienceQA translated to Thai | [CMKL/ScienceQA_TEST_TH](https://huggingface.co/datasets/CMKL/ScienceQA_TEST_TH) |
| `MMT-Bench_VAL_TH` | MMT-Bench translated to Thai | [CMKL/MMT-Bench_VAL_TH](https://huggingface.co/datasets/CMKL/MMT-Bench_VAL_TH) |
| `MathVista_TH` | MathVista translated to Thai | [CMKL/MathVista_TH](https://huggingface.co/datasets/CMKL/MathVista_TH) |
| `MathVerse_TH` | MathVerse translated to Thai | [CMKL/MathVerse_TH](https://huggingface.co/datasets/CMKL/MathVerse_TH) |
| `TGAT` | Thai General Aptitude Test | [CMKL/TGAT](https://huggingface.co/datasets/CMKL/TGAT) — see [dataset README](docs/TGAT_README.md) |
| `MathVista` | MathVista English (from HuggingFace) | [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista) |
| `MathVerse` | MathVerse English (from HuggingFace) | [AI4Math/MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse) |

---

## Configuration

Edit the YAML files in `configs/` to change the model endpoint or dataset list.

**`configs/standard.yaml`** — used by `run_eval.py` (standard benchmarks):
```yaml
eval_backend: VLMEvalKit
eval_config:
  model:
    - type: qwen3-omni
      name: CustomAPIModel
      api_base: http://localhost:8801/v1/chat/completions
      key: EMPTY
      temperature: 0.0
  data:
    - MMStar
    - SEEDBench_IMG
    - MMBench_DEV_EN
    - ScienceQA_TEST
```

**`configs/custom.yaml`** — used by `run_eval_custom.py` (Thai / custom benchmarks): same format with Thai dataset names.

> `type` must match the `--served-model-name` used when starting the vLLM server.

---

## Prerequisites

- Python 3.10+
- A running vLLM-served model endpoint (see [Step 0](#0-serve-the-model))
- HuggingFace access (public datasets, no token required)

---

## File Overview

```
evalscope-custom/
├── configs/
│   ├── standard.yaml       # Config for standard benchmark evaluation
│   └── custom.yaml         # Config for Thai/custom benchmark evaluation
├── docs/
│   ├── MMBench95CI_README.md
│   └── TGAT_README.md
├── LMUData/                # Raw downloaded datasets (TSV)
├── LMUData_formatted/      # Formatted datasets ready for evaluation
├── outputs/                # Evaluation results per run (named by timestamp)
├── custom_dataset.py       # Patches VLMEvalKit to load custom TSV datasets
├── download_datasets.py    # Downloads datasets from HuggingFace → LMUData/
├── format_eval.py          # Formats LMUData/ → evalscope-compatible TSV
├── run_eval.py             # Runs evaluation on standard benchmarks
├── run_eval_custom.py      # Runs evaluation on Thai/custom benchmarks
├── summary_report.py       # Generates accuracy summary CSV from outputs/
└── requirements.txt
```
