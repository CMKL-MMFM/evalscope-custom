# from custom_dataset import CustomDataset
from evalscope.run import run_task
import summary_report

run_task(task_cfg='config_template.yaml')

run_id = summary_report.latest_run_id()
if run_id:
    print(f"\n=== Auto-generating summary for run: {run_id} ===")
    summary_report.process_run(run_id)
else:
    print("No run output found — skipping summary report.")
