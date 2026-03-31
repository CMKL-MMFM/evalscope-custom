"""
summary_report.py
-----------------
Scan a VLMEvalKit output run and produce a CSV summary:
    dataset_name, overall_accuracy, questions_answered, questions_total

For datasets that failed evaluation (no _acc.csv produced), overall_accuracy
is left empty. questions_answered comes from the model result xlsx;
questions_total comes from the formatted TSV in LMUData_formated/.

Usage:
    python summary_report.py                        # latest run
    python summary_report.py 20260316_181212        # specific run id
    python summary_report.py --all                  # every run (one report each)
"""

import csv
import os
import sys

OUTPUTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
LMUDATA_FMT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LMUData_formated")


def get_overall(acc_csv_path: str) -> str:
    """Return the Overall value from an _acc.csv, or '' on failure."""
    try:
        with open(acc_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("Overall", "").strip().strip('"')
                if val:
                    return val
    except Exception:
        pass
    return ""


def count_answered(xlsx_path: str) -> int:
    """Count rows in the result xlsx where prediction is non-empty."""
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path, dtype=str)
        if "prediction" not in df.columns:
            return len(df)
        return int(df["prediction"].notna().sum())
    except Exception:
        return 0


def count_total(dataset_name: str) -> int:
    """Count rows in the formatted TSV for this dataset."""
    tsv_path = os.path.join(LMUDATA_FMT, dataset_name + ".tsv")
    if not os.path.isfile(tsv_path):
        # fallback: try the _TH variant (Thai-translated datasets)
        tsv_path = os.path.join(LMUDATA_FMT, dataset_name + "_TH.tsv")
    if not os.path.isfile(tsv_path):
        return 0
    try:
        with open(tsv_path, encoding="utf-8") as f:
            # subtract 1 for header
            return sum(1 for _ in f) - 1
    except Exception:
        return 0


def extract_dataset_name(filename: str, model_prefix: str) -> str:
    """
    Strip the model prefix and file suffix to get the dataset name.
    e.g. 'qwen2_5-omni_MMT-Bench_VAL_acc.csv' + 'qwen2_5-omni' → 'MMT-Bench_VAL'
         'qwen2_5-omni_MMBench_DEV_TH.xlsx'    + 'qwen2_5-omni' → 'MMBench_DEV_TH'
    """
    name = filename
    for ext in ("_acc.csv", ".xlsx", ".pkl"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    if name.endswith("_openai_result"):
        name = name[: -len("_openai_result")]
    prefix = model_prefix + "_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    return name


def collect_datasets(run_dir: str) -> dict:
    """
    Walk all model subdirectories (and their T*_G sub-subdirs) in a run_dir.
    Returns {dataset_name: {"accuracy": str, "answered": int, "total": int}}.
    T*_G results overwrite top-level ones (post-processed results preferred).
    """
    results: dict[str, dict] = {}

    for model_name in sorted(os.listdir(run_dir)):
        model_dir = os.path.join(run_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # search_dirs: model_dir first, then T*_G (so T*_G overwrites)
        search_dirs = [model_dir]
        for sub in os.listdir(model_dir):
            sub_path = os.path.join(model_dir, sub)
            if os.path.isdir(sub_path) and sub.startswith("T"):
                search_dirs.append(sub_path)

        for search_dir in search_dirs:
            acc_files: dict[str, str] = {}
            xlsx_files: dict[str, str] = {}

            for f in os.listdir(search_dir):
                if f.endswith("_acc.csv"):
                    ds = extract_dataset_name(f, model_name)
                    acc_files[ds] = os.path.join(search_dir, f)
                elif f.endswith(".xlsx") and not f.endswith("_openai_result.xlsx"):
                    ds = extract_dataset_name(f, model_name)
                    xlsx_files[ds] = os.path.join(search_dir, f)

            all_ds = set(xlsx_files) | set(acc_files)
            for ds in all_ds:
                accuracy = get_overall(acc_files[ds]) if ds in acc_files else ""
                answered = count_answered(xlsx_files[ds]) if ds in xlsx_files else 0
                total    = count_total(ds)
                results[ds] = {"accuracy": accuracy, "answered": answered, "total": total}

    return results


def write_summary(results: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_name", "overall_accuracy", "questions_answered", "questions_total"])
        for ds in sorted(results):
            r = results[ds]
            writer.writerow([ds, r["accuracy"], r["answered"] or "", r["total"] or ""])
    print(f"Summary written → {output_path}")

    # Console table
    print(f"\n{'dataset_name':<35} {'accuracy':<12} {'answered':>10} {'total':>8}")
    print("-" * 70)
    for ds in sorted(results):
        r = results[ds]
        acc      = r["accuracy"] if r["accuracy"] else "ERROR"
        answered = str(r["answered"]) if r["answered"] else "-"
        total    = str(r["total"])    if r["total"]    else "-"
        print(f"{ds:<35} {acc:<12} {answered:>10} {total:>8}")


def process_run(run_id: str):
    run_dir = os.path.join(OUTPUTS_DIR, run_id)
    if not os.path.isdir(run_dir):
        print(f"Run directory not found: {run_dir}")
        return
    results = collect_datasets(run_dir)
    if not results:
        print(f"No results found in {run_dir}")
        return
    output_path = os.path.join(run_dir, f"summary_{run_id}.csv")
    write_summary(results, output_path)


def latest_run_id() -> str | None:
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    runs = sorted(
        d for d in os.listdir(OUTPUTS_DIR)
        if os.path.isdir(os.path.join(OUTPUTS_DIR, d))
    )
    return runs[-1] if runs else None


def main():
    args = sys.argv[1:]

    if "--all" in args:
        if not os.path.isdir(OUTPUTS_DIR):
            print(f"outputs/ directory not found: {OUTPUTS_DIR}")
            return
        for run_id in sorted(os.listdir(OUTPUTS_DIR)):
            if os.path.isdir(os.path.join(OUTPUTS_DIR, run_id)):
                print(f"\n=== Run: {run_id} ===")
                process_run(run_id)
    elif args:
        process_run(args[0])
    else:
        run_id = latest_run_id()
        if not run_id:
            print(f"No run directories found in {OUTPUTS_DIR}")
            return
        print(f"Using latest run: {run_id}")
        process_run(run_id)


if __name__ == "__main__":
    main()
