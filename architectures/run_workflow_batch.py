"""
Batch runner for the workflow architecture across the full 22-task benchmark
set, matching run_custom_tasks.py's structure (same TASK_IDS, NUM_REPEATS,
MODEL, REPEAT_TEMPERATURE) so results are directly comparable to the
single-agent baseline in results/custom_baseline/.

Resumability matches run_custom_tasks.py: skips a (task_id, run_number) pair
if its output file already exists, so a partial/interrupted run can be
restarted without re-doing (and re-paying for) completed work.

Run with: python architectures/run_workflow_batch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures.workflow import run_workflow, save_result, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_LOG_DIR

# ---- Matches run_custom_tasks.py's TASK_IDS / NUM_REPEATS exactly ----
TASK_IDS = [
    722, 1942, 672, 1857, 122, 125, 792, 1030, 1089, 2056, 1463, 1563,
    89, 36, 446, 421, 334, 348, 1936, 1914, 1624, 1654,
]
NUM_REPEATS = 3
# ------------------------------------------------------------------


def main():
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

    for task_id in TASK_IDS:
        for run_number in range(1, NUM_REPEATS + 1):
            out_path = os.path.join(
                DEFAULT_LOG_DIR,
                f"results_{DEFAULT_MODEL}_workflow_task{task_id}_run{run_number}.json",
            )
            if os.path.exists(out_path):
                print(f"Skipping task_id={task_id} run={run_number} (output already exists: {out_path})")
                continue

            print(f"\n=== task_id={task_id} run {run_number}/{NUM_REPEATS} ===")
            result = run_workflow(
                task_id, run_number=run_number, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE
            )
            path = save_result(result, log_dir=DEFAULT_LOG_DIR, model=DEFAULT_MODEL)

            if result["agent_info"].get("source") == "api":
                print(f"CRASHED  task_id={task_id} run={run_number}  ({result['agent_info']['content']})")
            else:
                print(f"reward={result['reward']}  saved to {path}")


if __name__ == "__main__":
    main()
