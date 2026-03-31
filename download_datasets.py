"""
Download datasets from HuggingFace and save as TSV files into LMUData folder.
Add more datasets to DATASETS list as needed.
"""

import os
from datasets import load_dataset
import csv

# ── Add / remove datasets here ──────────────────────────────────────────────
# Each entry is either a plain "ORG/NAME" string (no config needed)
# or a tuple ("ORG/NAME", "config_name") for datasets that require one.
DATASETS = [
    # "CMKL/SEEDBench_IMG_TH",
    # "CMKL/ScienceQA_TEST_TH",
    # "CMKL/MMStar_TH",
    # "CMKL/MMT-Bench_VAL_TH",
    # "CMKL/MMBench_DEV_TH",
    # "CMKL/MathVerse_TH",
    # "CMKL/MathVista_TH",

    # "OpenGVLab/MMT-Bench", using only MMT-Bench_VAL.tsv
    # "AI4Math/MathVista",
    ("AI4Math/MathVerse", "testmini"),

    # "CMKL/TGAT",
    # "CMKL/MMBench95CI_DEV_TH"
]
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "LMUData"


def dataset_name_to_filename(name: str) -> str:
    """Convert 'ORG/DATASET_NAME' to 'DATASET_NAME.tsv'."""
    return name.split("/")[-1] + ".tsv"


def download_and_save(dataset_id: str, output_dir: str, config: str = None) -> None:
    filename = dataset_name_to_filename(dataset_id)
    output_path = os.path.join(output_dir, filename)

    if os.path.isfile(output_path):
        print(f"[{dataset_id}] Already exists at {output_path} — skipping.\n")
        return

    print(f"[{dataset_id}] Loading from HuggingFace...")
    ds = load_dataset(dataset_id, config) if config else load_dataset(dataset_id)

    # Use the first available split (usually 'train' or 'test')
    split_name = list(ds.keys())[0]
    split = ds[split_name]

    print(f"[{dataset_id}] Saving split '{split_name}' → {output_path}")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=split.column_names,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in split:
            writer.writerow(row)

    print(f"[{dataset_id}] Done — {len(split)} rows written.\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for entry in DATASETS:
        if isinstance(entry, tuple):
            dataset_id, config = entry
        else:
            dataset_id, config = entry, None
        try:
            download_and_save(dataset_id, OUTPUT_DIR, config)
        except Exception as e:
            print(f"[{dataset_id}] ERROR: {e}\n")

    print("All datasets processed.")


if __name__ == "__main__":
    main()
