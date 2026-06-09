# evalscope-custom

Evaluation pipeline for Vision-Language Models (VLMs) using [evalscope](https://github.com/modelscope/evalscope) / VLMEvalKit with Thai and custom datasets.

This repository includes custom Thai benchmark datasets and scripts to reproduce evaluation results from our paper.

---

## Datasets

This pipeline evaluates on the following datasets.

**Standard benchmarks** (auto-downloaded by VLMEvalKit):

| Dataset | Description |
|---------|-------------|
| `MMStar` | Multi-modal benchmark |
| `SEEDBench_IMG` | Image-based seed benchmark |
| `MMBench_DEV_EN` | MMBench English dev set |
| `ScienceQA_TEST` | Science QA test set |

**Thai / custom benchmarks** (downloaded and formatted by this repo):

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

## Prerequisites

- Python 3.10+
- A running vLLM-served model endpoint (see [Step 0](#0-serve-the-model))
- HuggingFace access (public datasets, no token required)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Steps

### 0. Serve the model

Start the model server before running evaluation. Example commands for supported models are in `serve_qwen3.txt` and `serve_qwen2.5.txt`.

**Qwen3-Omni (default):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit \
  --served-model-name qwen3-omni \
  --port 8801 \
  --trust-remote-code \
  --dtype float16 \
  --tensor-parallel-size 2
```

Update `api_base` in `config_custom.yaml` or `config_template.yaml` if you use a different port or model.

---

### 1. Download datasets

```bash
python download_datasets.py
```

Downloads Thai/custom datasets from HuggingFace and saves them as `.tsv` files into `LMUData/`.
To add or remove datasets, edit the `DATASETS` list at the top of `download_datasets.py`.

> Standard benchmarks (`MMStar`, `SEEDBench_IMG`, etc.) are downloaded automatically by VLMEvalKit during evaluation — you do not need to download them here.

---

### 2. Format datasets

```bash
python format_eval.py                              # all datasets
python format_eval.py MMStar_TH ScienceQA_TEST_TH  # specific datasets only
```

Converts raw TSV files from `LMUData/` into evalscope-compatible format and saves them to `LMUData_formated/`.
This step normalizes columns to `index | question | A–I | answer | image` and filters to multiple-choice questions only.

Some datasets (`MathVista`, `MathVerse`, `TGAT`) are loaded directly from HuggingFace during this step because their images require PIL decoding.

---

### 3. Run evaluation

**Standard benchmarks** (auto-downloaded by VLMEvalKit):
```bash
python run_eval.py
```
Uses `config_template.yaml`. Edit the `data` list in that file to select which benchmarks to run.

**Thai / custom benchmarks** (from `LMUData_formated/`):
```bash
python run_eval_custom.py
```
Uses `config_custom.yaml`. This script copies `LMUData_formated/` → `~/LMUData` so VLMEvalKit can locate the files, then starts evaluation.

Results are saved to `outputs/` and a summary CSV is auto-generated after each run.

---

### 4. View results

Results are saved under `outputs/<timestamp>/`. A per-dataset accuracy CSV is auto-generated at the end of each run.

To regenerate the summary manually:
```bash
python summary_report.py          # latest run only
python summary_report.py --all    # all runs in outputs/
```

---

## Configuration

Edit the YAML config files to change the model endpoint or dataset list before running.

**`config_template.yaml`** — used by `run_eval.py` (standard benchmarks):
```yaml
eval_backend: VLMEvalKit
eval_config:
  model:
    - type: qwen3-omni
      name: CustomAPIModel
      api_base: http://localhost:8801/v1/chat/completions  # change this if needed
      key: EMPTY
      temperature: 0.0
  data:
    - MMStar
    - SEEDBench_IMG
    # add or remove datasets here
```

**`config_custom.yaml`** — used by `run_eval_custom.py` (Thai / custom benchmarks): same format.

---

## File Overview

| File | Description |
|------|-------------|
| `download_datasets.py` | Downloads datasets from HuggingFace → `LMUData/` |
| `format_eval.py` | Formats `LMUData/` → evalscope-compatible TSV in `LMUData_formated/` |
| `run_eval.py` | Runs evaluation on standard benchmarks using `config_template.yaml` |
| `run_eval_custom.py` | Runs evaluation on Thai/custom benchmarks using `config_custom.yaml` |
| `custom_dataset.py` | Patches VLMEvalKit to load custom TSV datasets |
| `summary_report.py` | Generates accuracy summary CSV from `outputs/` |
| `config_template.yaml` | Config for standard benchmark evaluation |
| `config_custom.yaml` | Config for Thai/custom benchmark evaluation |
| `serve_qwen2.5.txt` | vLLM serve command for Qwen2.5 |
| `serve_qwen3.txt` | vLLM serve command for Qwen3-Omni |
| `LMUData/` | Raw downloaded datasets (TSV) |
| `LMUData_formated/` | Formatted datasets ready for evaluation |
| `outputs/` | Evaluation results per run (named by timestamp) |
| `docs/` | Dataset documentation |
