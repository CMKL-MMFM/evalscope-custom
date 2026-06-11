# from custom_dataset import CustomDataset
import os
from evalscope.run import run_task
import summary_report

run_task(task_cfg=os.path.join(os.path.dirname(__file__), "configs", "standard.yaml"))

run_id = summary_report.latest_run_id()
if run_id:
    print(f"\n=== Auto-generating summary for run: {run_id} ===")
    summary_report.process_run(run_id)
else:
    print("No run output found — skipping summary report.")
