import os
import shutil

from custom_dataset import CustomDataset  # noqa: F401 – patches CustomVQADataset
from evalscope.run import run_task
import summary_report

# ── Copy formatted datasets to ~/LMUData (where VLMEvalKit looks for them) ───
src = os.path.join(os.path.dirname(__file__), "LMUData_formated")
dst = os.path.expanduser("~/LMUData")

if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"Copied {src} → {dst}")

# ── Run evaluation ────────────────────────────────────────────────────────────
run_task(task_cfg=os.path.join(os.path.dirname(__file__), "config_custom.yaml"))

# ── Auto-generate summary report ─────────────────────────────────────────────
run_id = summary_report.latest_run_id()
if run_id:
    print(f"\n=== Auto-generating summary for run: {run_id} ===")
    summary_report.process_run(run_id)
else:
    print("No run output found — skipping summary report.")
