# MMBench95CI_DEV_TH — MMBench 95% CI Subset (Thai)

**HuggingFace:** [CMKL/MMBench95CI_DEV_TH](https://huggingface.co/datasets/CMKL/MMBench95CI_DEV_TH)

A Thai-translated subset of the MMBench development set, filtered to questions with statistically robust accuracy estimates. The subset contains items selected based on a 95% confidence interval criterion to reduce noise in benchmark comparisons.

---

## Dataset Description

| Property | Value |
|----------|-------|
| Language | Thai (questions and choices) |
| Base dataset | [MMBench](https://github.com/open-compass/MMBench) DEV split |
| Question type | Multiple-choice (2–4 options: A–D) |
| Modality | Image + text |
| Answer | Single letter (A, B, C, or D) |

---

## Format

Each example contains:

| Field | Description |
|-------|-------------|
| `index` | Unique question identifier |
| `image` | Base64-encoded image |
| `answer` | Correct answer letter (A–D) |
| `question` | Original English question text |
| `A`, `B`, `C`, `D` | Original English answer choices |
| `question_th` | Thai-translated question text |
| `A_th`, `B_th`, `C_th`, `D_th` | Thai-translated answer choices (C_th / D_th may be empty for 2-option questions) |

---

## How It Is Used in This Pipeline

`format_eval.py` processes this dataset from the downloaded TSV (`LMUData/MMBench95CI_DEV_TH.tsv`):

1. Uses the Thai columns (`question_th`, `A_th`–`D_th`) for the question and choices.
2. Keeps the original `answer` letter (A–D) unchanged.
3. Handles missing C/D options (2-option questions have empty C_th/D_th).
4. Outputs `LMUData_formated/MMBench95CI_DEV_TH.tsv`.

To download then format:
```bash
python download_datasets.py   # downloads MMBench95CI_DEV_TH.tsv into LMUData/
python format_eval.py MMBench95CI_DEV_TH
```

To run evaluation on this dataset, ensure `MMBench95CI_DEV_TH` is listed under `data` in `config_custom.yaml`, then:
```bash
python run_eval_custom.py
```

---

## Notes

- The 95% CI subset is a filtered version of MMBench DEV — not all items from the full dev split are included.
- Evaluation uses Thai question and choice text; the correct answer label (A–D) is language-independent.
- This dataset is part of a broader effort to evaluate Thai-language VLM capabilities on established visual reasoning benchmarks.
