# TGAT — Thai General Aptitude Test

**HuggingFace:** [CMKL/TGAT](https://huggingface.co/datasets/CMKL/TGAT)

A visual multiple-choice benchmark based on the Thai General Aptitude Test (TGAT), a standardized university entrance exam in Thailand. This dataset evaluates reasoning, spatial understanding, and general aptitude through image-based questions.

---

## Dataset Description

| Property | Value |
|----------|-------|
| Language | Thai |
| Question type | Multiple-choice (5 options: A–E) |
| Modality | Image (question text embedded in image) |
| Answer type | Single correct answer |
| HuggingFace split | `train` |

Questions and answer choices are embedded directly in the image — there is no separate text field for the question body. The model must read and reason over the image to select the correct letter.

---

## Format

Each example contains:

| Field | Description |
|-------|-------------|
| `id` | Unique question identifier |
| `image` | PIL Image containing the full question and answer choices |
| `answers` | Dict mapping each letter (`A`–`E`) to a score (`1.0` = correct, `0.0` = incorrect) |
| `answer_type` | `"single"` (one correct answer), `"multi"`, or `"soft"` |

Only `answer_type == "single"` items are used in evaluation.

---

## How It Is Used in This Pipeline

`format_eval.py` processes TGAT directly from HuggingFace (no prior `download_datasets.py` step needed for TGAT):

1. Loads `CMKL/TGAT` split `train` from HuggingFace.
2. Filters to `answer_type == "single"` rows only.
3. Converts each PIL image to a compressed JPEG base64 string.
4. Sets choice columns (`A`–`E`) to their letter labels as placeholders (choices are embedded in the image, not text fields).
5. Extracts the correct letter from the `answers` dict (the key with score `1.0`).
6. Outputs `LMUData_formated/TGAT.tsv`.

The model is prompted in Thai:
> "โปรดตอบคำถามโดยตรงและระบุตัวเลือกที่ถูกต้องเป็นตัวอักษร เช่น A, B, C, D หรือ E เท่านั้น"
> *(Please answer directly and indicate only the correct answer as a letter: A, B, C, D, or E.)*

To format TGAT only:
```bash
python format_eval.py TGAT
```

---

## Notes

- Questions with `answer_type` other than `"single"` (multi-answer or partial-credit) are excluded.
- Questions where the correct letter cannot be determined from `answers` are also excluded.
- Because choices are visual (inside the image), the choice text columns in the TSV contain only the letter itself (`A`, `B`, `C`, `D`, `E`) as a non-empty placeholder to satisfy VLMEvalKit's internal format.
