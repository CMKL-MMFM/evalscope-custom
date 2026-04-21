# evalscope-custom

Evaluation pipeline for VLMs using VLMEvalKit / evalscope with Thai and custom datasets.

---

## Steps

### 1. Download datasets

```bash
python download_datasets.py
```

Downloads datasets from HuggingFace and saves them as `.tsv` files into `LMUData/`.
To add or remove datasets, edit the `DATASETS` list at the top of the file.

---

### 2. Format datasets

```bash
python format_eval.py                        # all datasets
python format_eval.py MMStar_TH ScienceQA_TEST_TH   # specific datasets only
```

Converts raw TSV files from `LMUData/` into evalscope-compatible format and saves them to `LMUData_formated/`.
This step normalizes columns (`index`, `question`, `A`–`I`, `answer`, `image`) and filters to multiple-choice questions only.

---

### 3. Run evaluation

**Original datasets** (downloaded directly by evalscope/VLMEvalKit):
```bash
python run_eval.py
```
Uses `config_template.yaml`. Datasets are standard benchmarks like `MMStar`, `SEEDBench_IMG`, etc.

**Custom / Thai datasets** (from `LMUData_formated/`):
```bash
python run_eval_custom.py
```
Uses `config_custom.yaml`. Copies `LMUData_formated/` → `~/LMUData` before running so VLMEvalKit can find them.

Results are saved to `outputs/` and a summary CSV is auto-generated after each run.

---

## Other files

| File | Description |
|------|-------------|
| `config_template.yaml` | Config for `run_eval.py` — set model endpoint, datasets, and options |
| `config_custom.yaml` | Config for `run_eval_custom.py` — same format, points to custom/Thai datasets |
| `custom_dataset.py` | Patches VLMEvalKit's `CustomVQADataset` to load and prompt custom TSV datasets |
| `summary_report.py` | Scans `outputs/` and generates a CSV with accuracy per dataset. Run manually with `python summary_report.py` or `python summary_report.py --all` |
| `serve_qwen2.5.txt` | Command to serve Qwen2.5 model locally via vLLM |
| `serve_qwen3.txt` | Command to serve Qwen3-Omni model locally via vLLM |
| `LMUData/` | Raw downloaded datasets (TSV) |
| `LMUData_formated/` | Formatted datasets ready for evaluation |
| `outputs/` | Evaluation results, one folder per run (named by timestamp) |
