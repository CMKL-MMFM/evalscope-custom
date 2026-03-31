"""
format_eval.py
--------------
Convert raw datasets in LMUData/ → evalscope-compatible TSV files in LMUData_formated/

Output columns (required by CustomVQADataset / VLMEvalKit):
    index | question | A | B | C | D | answer | image

Dataset notes
─────────────
From TSV (LMUData/), images already raw base64:
  SEEDBench_IMG_TH  – standard A/B/C/D cols, answer letter
  MMStar_TH         – standard A/B/C/D cols, answer letter
  MMBench_DEV_TH    – standard A/B/C/D cols, answer letter, C/D may be nan
  MMT-Bench_VAL_TH  – A/B/C/D cols (E–I ignored), skip rows where answer > D
  MMT-Bench_VAL     – same as above (English)
  ScienceQA_TEST_TH – filter task=="closed choice", A/B/C/D cols, answer letter
  MathVista_TH      – filter question_type=="multi_choice", choices list → A/B/C/D,
                       answer_option is the letter
  MathVerse_TH      – filter question_type=="multi-choice", parse A/B/C/D from
                       query_wo "Choices:\\nA:...\\nB:..." block, answer letter

From HuggingFace (images need PIL decode):
  MathVista         – AI4Math/MathVista  split=testmini, decoded_image PIL → base64,
                       filter multi_choice, choices list, match answer text → letter
  MathVerse         – AI4Math/MathVerse  config=testmini, image PIL → base64,
                       filter multi-choice, parse A/B/C/D from query_wo, answer letter

Usage:
    python format_eval.py                  # all datasets
    python format_eval.py ScienceQA_TEST_TH MMStar_TH   # specific datasets by stem
"""

import ast
import io
import os
import re
import sys
import base64

import pandas as pd

LMUDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LMUData")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LMUData_formated")
STANDARD_HEADER = "index\tquestion\tA\tB\tC\tD\tanswer\timage\n"
LETTERS = ["A", "B", "C", "D"]

# pandas C parser has a ~1 MB per-line limit; keep images well under that.
# 400 KB image → ~533 K base64 chars → safe with surrounding text.
MAX_B64_CHARS = 400_000
IMG_MAX_DIM   = 768   # resize longest edge to this before re-encoding
IMG_QUALITY   = 82    # JPEG quality


# ── Utilities ──────────────────────────────────────────────────────────────────

def clean(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).replace("\t", " ").replace("\n", " ").replace("\r", "").strip()


def _pil_compress(img, max_dim=IMG_MAX_DIM, quality=IMG_QUALITY) -> str:
    """Resize PIL image to max_dim and encode as JPEG base64."""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_b64(img) -> str:
    """Convert PIL image → compressed JPEG base64."""
    return _pil_compress(img)


def compress_b64(b64: str) -> str:
    """
    If a raw base64 image string exceeds MAX_B64_CHARS, decode → PIL → re-encode
    at reduced size.  Returns the (possibly compressed) base64 string.
    """
    if not isinstance(b64, str) or len(b64) <= MAX_B64_CHARS:
        return b64
    try:
        from PIL import Image
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        return _pil_compress(img)
    except Exception:
        # If we can't decompress, truncate is worse than returning original;
        # just return as-is and let VLMEvalKit deal with it.
        return b64


def parse_choices_list(raw) -> list:
    """Parse a Python-list string like "['opt1','opt2',...]" → list of str."""
    if not isinstance(raw, str) or raw.strip() in ("nan", ""):
        return []
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, list):
            return [str(x) for x in result]
    except Exception:
        pass
    return []


# Thai alphabet option letters → A/B/C/D
THAI_LETTERS = ["ก", "ข", "ค", "ง"]
THAI_TO_LATIN = {t: l for t, l in zip(THAI_LETTERS, LETTERS)}


def parse_choices_from_query_wo(query_wo: str) -> dict:
    """
    Extract choices from MathVerse query_wo.

    Handles both English and Thai formats:
        Choices:          or   ตัวเลือก:
        A:40°                  ก: 40°
        B:60°                  ข: 60°
        C:120°                 ค: 120°
        D:140°                 ง: 140°

    Always returns {'A': '...', 'B': '...', ...} with Latin keys.
    """
    text = str(query_wo)
    match = re.search(r'(?:Choices|ตัวเลือก)\s*:\s*\n(.*?)$', text, re.DOTALL)
    if not match:
        return {}
    result = {}
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        # Match Latin A-D or Thai ก-ง, followed by : or ：
        m = re.match(r'^([A-Dก-ง])\s*[:：]\s*(.+)$', line)
        if m:
            key = m.group(1)
            key = THAI_TO_LATIN.get(key, key)  # normalise Thai → Latin
            result[key] = m.group(2).strip()
    return result


def answer_to_letter(raw) -> str:
    """Return 'A'/'B'/'C'/'D' from int index, Latin letter, or Thai letter."""
    if isinstance(raw, int) and 0 <= raw < 4:
        return LETTERS[raw]
    s = str(raw).strip()
    if s in THAI_TO_LATIN:
        return THAI_TO_LATIN[s]
    s = s.upper()
    if s in LETTERS:
        return s
    return s  # keep as-is; caller decides whether to skip


def write_rows(rows, output_path, extra_cols: list[str] | None = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extra_cols = extra_cols or []
    header = STANDARD_HEADER.rstrip("\n")
    if extra_cols:
        header += "\t" + "\t".join(extra_cols)
    header += "\n"
    compressed = 0
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        for r in rows:
            img = r["image"]
            new_img = compress_b64(img)
            if new_img is not img:
                compressed += 1
            # Strip any newlines/tabs from base64 image data to keep TSV well-formed
            new_img = new_img.replace("\n", "").replace("\r", "").replace("\t", "")
            line = (
                f"{r['index']}\t{r['question']}\t{r['A']}\t{r['B']}\t"
                f"{r['C']}\t{r['D']}\t{r['answer']}\t{new_img}"
            )
            for col in extra_cols:
                line += "\t" + clean(r.get(col, ""))
            line += "\n"
            f.write(line)
    if compressed:
        print(f"  🗜️   Compressed {compressed} oversized images")
    print(f"  ✅  {len(rows)} rows → {output_path}")


# ── Per-dataset formatters (read from LMUData TSV) ────────────────────────────

def fmt_standard_abcd(df: pd.DataFrame) -> list:
    """
    Datasets with columns: index, question, A, B, C, D, answer (letter), image.
    SEEDBench_IMG_TH, MMStar_TH
    """
    rows = []
    for idx, row in enumerate(df.itertuples(index=False), 1):
        rows.append({
            "index":   getattr(row, "index", idx),
            "question": clean(row.question),
            "A":        clean(row.A),
            "B":        clean(row.B),
            "C":        clean(row.C),
            "D":        clean(row.D),
            "answer":   answer_to_letter(row.answer),
            "image":    clean(row.image),
        })
    return rows


def fmt_mmbench(df: pd.DataFrame) -> list:
    """
    MMBench_DEV_TH: index, answer, category, image, hint, question, A, B, C, D
    C and D may be NaN (2-option questions).
    """
    rows = []
    for idx, row in enumerate(df.itertuples(index=False), 1):
        rows.append({
            "index":   getattr(row, "index", idx),
            "question": clean(row.question),
            "A":        clean(row.A),
            "B":        clean(row.B),
            "C":        clean(row.C),
            "D":        clean(row.D),
            "answer":   answer_to_letter(row.answer),
            "image":    clean(row.image),
        })
    return rows


def fmt_mmbench95ci(df: pd.DataFrame) -> list:
    """
    MMBench95CI_DEV_TH: same layout as MMBench_DEV_TH but has additional
    Thai columns: question_th, A_th, B_th, C_th, D_th.
    Uses Thai columns for question/choices; C_th/D_th may be NaN.
    """
    rows = []
    for idx, row in enumerate(df.itertuples(index=False), 1):
        rows.append({
            "index":   getattr(row, "index", idx),
            "question": clean(row.question_th),
            "A":        clean(row.A_th),
            "B":        clean(row.B_th),
            "C":        clean(getattr(row, "C_th", "")),
            "D":        clean(getattr(row, "D_th", "")),
            "answer":   answer_to_letter(row.answer),
            "image":    clean(row.image),
        })
    return rows


def fmt_mmtbench(df: pd.DataFrame, question_generated: str = "") -> list:
    """
    MMT-Bench_VAL_TH / MMT-Bench_VAL:
    Has columns A–I but evalscope only handles A–D.
    Skip rows where the correct answer is E or beyond.

    question_generated: fixed MCQ prompt to replace the original question field.
    The original question text is preserved in the 'question_generated' extra column.
    """
    rows = []
    skipped = 0
    for idx, row in enumerate(df.itertuples(index=False), 1):
        letter = answer_to_letter(row.answer)
        if letter not in LETTERS:
            skipped += 1
            continue
        original_question = clean(row.question)
        rows.append({
            "index":              getattr(row, "index", idx),
            "question":           question_generated if question_generated else original_question,
            "A":                  clean(row.A),
            "B":                  clean(row.B),
            "C":                  clean(row.C),
            "D":                  clean(row.D),
            "answer":             letter,
            "image":              clean(row.image),
            "question_generated": original_question,
        })
    if skipped:
        print(f"    ⚠️  Skipped {skipped} rows with answer outside A–D")
    return rows


def fmt_sciqa(df: pd.DataFrame) -> list:
    """
    ScienceQA_TEST_TH: filter task=="closed choice".
    Columns: question, answer, image, task, A, B, C, D (E rare, ignored)
    """
    rows = []
    for idx, row in enumerate(df.itertuples(index=False), 1):
        if clean(row.task) != "closed choice":
            continue
        rows.append({
            "index":   getattr(row, "index", idx),
            "question": clean(row.question),
            "A":        clean(row.A),
            "B":        clean(row.B),
            "C":        clean(row.C),
            "D":        clean(row.D),
            "answer":   answer_to_letter(row.answer),
            "image":    clean(row.image),
        })
    return rows


def fmt_mathvista_th(df: pd.DataFrame) -> list:
    """
    MathVista_TH:
    - Filter question_type == "multi_choice"
    - choices is a Python list string: "['opt1','opt2','opt3','opt4']"
    - answer_option is the letter (A/B/C/D)
    - image is raw base64
    """
    rows = []
    skipped = 0
    for idx, row in enumerate(df.itertuples(index=False), 1):
        if clean(row.question_type) != "multi_choice":
            continue
        choices = parse_choices_list(row.choices)
        if len(choices) < 2:
            skipped += 1
            continue
        while len(choices) < 4:
            choices.append("")
        letter = answer_to_letter(row.answer_option)
        if letter not in LETTERS:
            skipped += 1
            continue
        rows.append({
            "index":   getattr(row, "index", idx),
            "question": clean(row.question),
            "A":        clean(choices[0]),
            "B":        clean(choices[1]),
            "C":        clean(choices[2]),
            "D":        clean(choices[3]),
            "answer":   letter,
            "image":    clean(row.image),
        })
    if skipped:
        print(f"    ⚠️  Skipped {skipped} non-MCQ or bad rows")
    return rows


def fmt_mathverse_th(df: pd.DataFrame) -> list:
    """
    MathVerse_TH:
    - Filter question_type == "multi-choice"
    - No A/B/C/D columns; parse choices from query_wo: "Choices:\\nA:...\\nB:..."
    - answer is already a letter
    - image is raw base64
    """
    rows = []
    skipped = 0
    for idx, row in enumerate(df.itertuples(index=False), 1):
        if clean(row.question_type) != "multi-choice":
            continue
        choices = parse_choices_from_query_wo(row.query_wo)
        if len(choices) < 2:
            skipped += 1
            continue
        letter = answer_to_letter(row.answer)
        if letter not in LETTERS:
            skipped += 1
            continue
        rows.append({
            "index":   getattr(row, "index", idx),
            "question": clean(row.question),
            "A":        choices.get("A", ""),
            "B":        choices.get("B", ""),
            "C":        choices.get("C", ""),
            "D":        choices.get("D", ""),
            "answer":   letter,
            "image":    clean(row.image),
        })
    if skipped:
        print(f"    ⚠️  Skipped {skipped} non-MCQ or unparseable rows")
    return rows


# ── HuggingFace formatter (needs PIL image decode) ────────────────────────────

def fmt_mathvista_hf(output_path: str):
    """
    AI4Math/MathVista (English original):
    - Loads from HuggingFace (split=testmini)
    - decoded_image is a PIL Image → convert to base64
    - Filter question_type == "multi_choice"
    - choices is a Python list, answer is the text → match to A/B/C/D
    """
    from datasets import load_dataset

    print("  Loading AI4Math/MathVista from HuggingFace (split=testmini)...")
    ds = load_dataset("AI4Math/MathVista", split="testmini")

    rows = []
    skipped = 0
    for idx, item in enumerate(ds, 1):
        if item.get("question_type") != "multi_choice":
            continue
        choices_raw = item.get("choices", [])
        if isinstance(choices_raw, str):
            choices_raw = parse_choices_list(choices_raw)
        choices = [str(c) for c in (choices_raw or [])]
        if len(choices) < 2:
            skipped += 1
            continue
        while len(choices) < 4:
            choices.append("")

        answer_text = str(item.get("answer", "")).strip()
        letter = "A"
        for i, ch in enumerate(choices[:4]):
            if ch.strip() == answer_text:
                letter = LETTERS[i]
                break
        else:
            if answer_text.upper() in LETTERS:
                letter = answer_text.upper()
            else:
                skipped += 1
                continue

        img = item.get("decoded_image") or item.get("image")
        if hasattr(img, "save"):
            img_b64 = pil_to_b64(img)
        elif isinstance(img, bytes):
            img_b64 = base64.b64encode(img).decode()
        elif isinstance(img, str) and not img.startswith("images/"):
            img_b64 = img  # already base64
        else:
            img_b64 = ""

        rows.append({
            "index":    item.get("pid", idx),
            "question": clean(item.get("question", "")),
            "A":        clean(choices[0]),
            "B":        clean(choices[1]),
            "C":        clean(choices[2]),
            "D":        clean(choices[3]),
            "answer":   letter,
            "image":    img_b64,
        })

    if skipped:
        print(f"    ⚠️  Skipped {skipped} non-MCQ or unmatched rows")
    write_rows(rows, output_path)


def fmt_mathverse_hf(output_path: str):
    """
    AI4Math/MathVerse (English original, testmini split):
    - Loads from HuggingFace (config=testmini)
    - Filter question_type == "multi-choice"
    - Parse A/B/C/D choices from query_wo "Choices:\\nA:...\\nB:..." block
    - answer is already a letter
    - image is a PIL Image → convert to base64
    """
    from datasets import load_dataset

    print("  Loading AI4Math/MathVerse from HuggingFace (config=testmini)...")
    ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")

    rows = []
    skipped = 0
    for idx, item in enumerate(ds, 1):
        if item.get("question_type") != "multi-choice":
            skipped += 1
            continue
        choices = parse_choices_from_query_wo(item.get("query_wo", ""))
        if len(choices) < 2:
            skipped += 1
            continue
        letter = answer_to_letter(item.get("answer", ""))
        if letter not in LETTERS:
            skipped += 1
            continue

        img = item.get("image")
        if hasattr(img, "save"):
            img_b64 = pil_to_b64(img)
        elif isinstance(img, bytes):
            img_b64 = base64.b64encode(img).decode()
        elif isinstance(img, str):
            img_b64 = img
        else:
            img_b64 = ""

        rows.append({
            "index":    item.get("problem_index", idx),
            "question": clean(item.get("question", "")),
            "A":        choices.get("A", ""),
            "B":        choices.get("B", ""),
            "C":        choices.get("C", ""),
            "D":        choices.get("D", ""),
            "answer":   letter,
            "image":    img_b64,
        })

    if skipped:
        print(f"    ⚠️  Skipped {skipped} non-MCQ or unparseable rows")
    write_rows(rows, output_path)


def fmt_tgat_hf(output_path: str):
    """
    CMKL/TGAT (Thai General Aptitude Test):
    - Loads from HuggingFace (split=train)
    - image is a PIL Image → convert to base64
    - Filter answer_type == "single" (skip multi / soft)
    - answers is a dict {letter: score}; 1.0 = correct
    - Skip rows where the correct answer is E (only A–D supported)
    - No question text; question is embedded in the image
    """
    from datasets import load_dataset

    print("  Loading CMKL/TGAT from HuggingFace...")
    ds = load_dataset("CMKL/TGAT", split="train")

    q_gen = "โปรดตอบคำถามโดยตรงและระบุตัวเลือกที่ถูกต้องเป็นตัวอักษร เช่น A, B, C, D"

    rows = []
    skipped = 0
    for idx, item in enumerate(ds, 1):
        if item.get("answer_type") != "single":
            skipped += 1
            continue

        answers = item.get("answers", {})
        letter = None
        for k, v in answers.items():
            try:
                if float(v) == 1.0:
                    letter = str(k).strip().upper()
                    break
            except (TypeError, ValueError):
                pass
        if letter is None or letter not in LETTERS:
            skipped += 1
            continue

        img = item.get("image")
        if hasattr(img, "save"):
            img_b64 = pil_to_b64(img)
        elif isinstance(img, bytes):
            img_b64 = base64.b64encode(img).decode()
        elif isinstance(img, str):
            img_b64 = img
        else:
            img_b64 = ""

        rows.append({
            "index":   item.get("id", idx),
            "question": q_gen,
            "A":        "",
            "B":        "",
            "C":        "",
            "D":        "",
            "answer":   letter,
            "image":    img_b64,
        })

    if skipped:
        print(f"    ⚠️  Skipped {skipped} non-single or answer-E rows")
    write_rows(rows, output_path)


# ── Dataset registry ───────────────────────────────────────────────────────────
# (tsv_filename, formatter_fn)  — formatter_fn receives a pd.DataFrame
# For HuggingFace-sourced datasets, tsv_filename is None and hf_fn is used.

_Q_MMT_TH = "โปรดเลือกคำตอบที่ดีที่สุดจาก Choice A, B, C, D"
_Q_MMT_EN = "Please choose the best answer from Choices A, B, C, and D."

TSV_CONFIGS = [
    ("SEEDBench_IMG_TH.tsv",   fmt_standard_abcd),
    ("MMStar_TH.tsv",          fmt_standard_abcd),
    ("MMBench_DEV_TH.tsv",     fmt_mmbench),
    ("MMBench95CI_DEV_TH.tsv", fmt_mmbench95ci),
    ("MMT-Bench_VAL_TH.tsv",   lambda df: fmt_mmtbench(df, _Q_MMT_TH)),
    ("MMT-Bench_VAL.tsv",      lambda df: fmt_mmtbench(df, _Q_MMT_EN)),
    ("ScienceQA_TEST_TH.tsv",  fmt_sciqa),
    ("MathVista_TH.tsv",       fmt_mathvista_th),
    ("MathVerse_TH.tsv",       fmt_mathverse_th),
]

HF_CONFIGS = [
    # (output_filename, hf_formatter_fn)
    ("MathVista.tsv",  fmt_mathvista_hf),
    ("MathVerse.tsv",  fmt_mathverse_hf),
    ("TGAT.tsv",       fmt_tgat_hf),
]


# ── Main ───────────────────────────────────────────────────────────────────────

def process_tsv(tsv_file, formatter):
    src = os.path.join(LMUDATA_DIR, tsv_file)
    if not os.path.isfile(src):
        print(f"  ⚠️  {tsv_file} not found in LMUData/ — skipping")
        return
    print(f"\n[{tsv_file}] Reading {src} ...")
    df = pd.read_csv(src, sep="\t", dtype=str, on_bad_lines="skip")
    print(f"  {len(df)} rows, columns: {list(df.columns)}")
    try:
        rows = formatter(df)
    except Exception as e:
        print(f"  ⚠️  Formatter error: {e}")
        raise
    if not rows:
        print(f"  ⚠️  No rows produced — skipping output")
        return
    extra_cols = [k for k in rows[0] if k not in ("index", "question", "A", "B", "C", "D", "answer", "image")]
    write_rows(rows, os.path.join(OUTPUT_DIR, tsv_file), extra_cols=extra_cols or None)


def process_hf(output_file, hf_fn):
    output_path = os.path.join(OUTPUT_DIR, output_file)
    print(f"\n[{output_file}] Loading from HuggingFace ...")
    try:
        hf_fn(output_path)
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        raise


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    targets = set(sys.argv[1:])

    def wanted(name):
        if not targets:
            return True
        stem = name.replace(".tsv", "")
        return stem in targets or name in targets

    for tsv_file, formatter in TSV_CONFIGS:
        if wanted(tsv_file):
            process_tsv(tsv_file, formatter)

    for output_file, hf_fn in HF_CONFIGS:
        if wanted(output_file):
            process_hf(output_file, hf_fn)

    print(f"\nDone. Formatted files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
