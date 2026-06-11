# MMBench95CI_DEV_TH — MMBench 95% CI Subset (Thai)

**HuggingFace:** [CMKL/MMBench95CI_DEV_TH](https://huggingface.co/datasets/CMKL/MMBench95CI_DEV_TH)

A high-quality Thai benchmark derived from the MMBench development set through a rigorous three-stage process: AI-assisted translation, human expert validation, and statistical filtering. The result is a carefully curated subset of **353 samples** that provides statistically reliable accuracy estimates for Thai-language VLM evaluation.

---

## Dataset Description

| Property | Value |
|----------|-------|
| Language | Thai (questions and choices) |
| Base dataset | [MMBench](https://github.com/open-compass/MMBench) DEV split |
| Question type | Multiple-choice (2–4 options: A–D) |
| Modality | Image + text |
| Answer | Single letter (A, B, C, or D) |
| Final sample size | 353 questions |

---

## Dataset Construction

This dataset was built from the full MMBench DEV split through the following pipeline:

**Step 1 — Statistical Filtering (95% Confidence Interval)**
Rather than translating and validating the entire MMBench DEV split — which would have required more human resources than were available — we first applied a 95% confidence interval criterion to identify the most statistically informative questions. Items where model accuracy estimates were too noisy (questions that are near-universally correct or near-random) were removed. This filtering reduced the dataset to a focused subset of **353 samples**, ensuring that every remaining item meaningfully discriminates between models while keeping the human annotation workload tractable.

**Step 2 — AI Translation**
The 353 filtered questions and their answer choices were translated from English to Thai using an AI translation model, producing an initial Thai draft.

**Step 3 — Human Validation and Adjustment**
Native Thai-speaking annotators reviewed every translated item. Translations that were unnatural, ambiguous, or technically imprecise were manually corrected. This step ensured that the Thai text faithfully preserved the original meaning and would be clearly interpretable by both humans and models.

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
| `question_th` | Human-validated Thai question text |
| `A_th`, `B_th`, `C_th`, `D_th` | Human-validated Thai answer choices (C_th / D_th may be empty for 2-option questions) |

---

## How It Is Used in This Pipeline

`format_eval.py` processes this dataset from the downloaded TSV (`LMUData/MMBench95CI_DEV_TH.tsv`):

1. Uses the Thai columns (`question_th`, `A_th`–`D_th`) for the question and choices.
2. Keeps the original `answer` letter (A–D) unchanged.
3. Handles missing C/D options (2-option questions have empty C_th/D_th).
4. Outputs `LMUData_formatted/MMBench95CI_DEV_TH.tsv`.

To download then format:
```bash
python download_datasets.py   # downloads MMBench95CI_DEV_TH.tsv into LMUData/
python format_eval.py MMBench95CI_DEV_TH
```

To run evaluation on this dataset, ensure `MMBench95CI_DEV_TH` is listed under `data` in `configs/custom.yaml`, then:
```bash
python run_eval_custom.py
```

---

## Notes

- The 353-sample subset is a statistically filtered version of MMBench DEV — not all items from the full dev split are included.
- Evaluation uses Thai question and choice text; the correct answer label (A–D) is language-independent.
- This dataset is part of a broader effort to evaluate Thai-language VLM capabilities on established visual reasoning benchmarks.
