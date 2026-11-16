"""
Batch runner for the multi-agent supervisor architecture across the full 24-task
benchmark set (the 22 real CRMArena-Pro tasks + custom tasks 900001/900002),
mirroring run_workflow_batch.py's structure (same MODEL, REPEAT_TEMPERATURE via
multi_agent.py's defaults, which re-use workflow.py's) so results are directly
comparable to results/workflow/ and results/custom_baseline/.

Resumability matches run_workflow_batch.py: skips a (task_id, run_number) pair if
its output file already exists -- with one deliberate tightening, since "skip
what's already valid" is the point: an existing file is only skipped if it is not
an infra-error record (agent_info.source == "api", the same signal CLAUDE.md's
"Known gotchas" documents for a crashed/rate-limited run silently saved as
reward=0). A crashed run's file is re-run and overwritten rather than being
treated as completed work forever.

`import custom_tasks` is required for task_ids 900001/900002: multi_agent.py's
_load_task looks tasks up in TASKS_B2C, which custom_tasks.py extends in place.

Run with: python architectures/run_multi_agent_batch.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import custom_tasks  # noqa: F401 -- side effect: extends TASKS_B2C with 900001/900002
from architectures.multi_agent import (
    run_multi_agent, save_result, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_LOG_DIR,
)

# ---- Same 22 real task_ids as run_workflow_batch.py / run_custom_tasks.py,
# ---- plus the 2 custom tasks.
TASK_IDS = [
    722, 1942, 672, 1857, 122, 125, 792, 1030, 1089, 2056, 1463, 1563,
    89, 36, 446, 421, 334, 348, 1936, 1914, 1624, 1654,
    900001, 900002,
]
NUM_REPEATS = 3
# ------------------------------------------------------------------


def _out_path(task_id, run_number):
    return os.path.join(
        DEFAULT_LOG_DIR,
        f"results_{DEFAULT_MODEL}_multi_agent_task{task_id}_run{run_number}.json",
    )


def _is_valid_existing(path):
    """True if `path` exists and is a completed (non-infra-error) result."""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            result = json.load(f)[0]
    except (json.JSONDecodeError, IndexError, OSError):
        return False
    return result.get("agent_info", {}).get("source") != "api"


def pending_runs():
    return [
        (task_id, run_number)
        for task_id in TASK_IDS
        for run_number in range(1, NUM_REPEATS + 1)
        if not _is_valid_existing(_out_path(task_id, run_number))
    ]


def main():
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

    for task_id in TASK_IDS:
        for run_number in range(1, NUM_REPEATS + 1):
            out_path = _out_path(task_id, run_number)
            if _is_valid_existing(out_path):
                print(f"Skipping task_id={task_id} run={run_number} (valid output already exists: {out_path})")
                continue

            print(f"\n=== task_id={task_id} run {run_number}/{NUM_REPEATS} ===", flush=True)
            result = run_multi_agent(
                task_id, run_number=run_number, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE
            )
            path = save_result(result, log_dir=DEFAULT_LOG_DIR, model=DEFAULT_MODEL)

            if result["agent_info"].get("source") == "api":
                print(f"CRASHED  task_id={task_id} run={run_number}  ({result['agent_info']['content']})", flush=True)
            else:
                print(f"reward={result['reward']}  saved to {path}", flush=True)


if __name__ == "__main__":
    main()
