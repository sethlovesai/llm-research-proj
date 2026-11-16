"""
Validation run for architectures/workflow.py on the 3 validated decomposition
examples (task_ids 722, 2056, 89 -- see decomposition/validate_decompose.py).

Prints the full execution trace per task (order, context passed between
subtasks, each subtask's output) and the final scored result, and saves
results under results/workflow/ in the same schema as results/custom_baseline/.

Always uses normalize_score.py to report pass/fail -- per CLAUDE.md's
documented gotcha, a raw `reward == 1` check silently misreports fuzzy_match
tasks (reward is a dict there, never == 1).

Run with: python architectures/test_workflow.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures.workflow import run_workflow, save_result, DEFAULT_MODEL
from normalize_score import normalize_score

TASK_IDS = [722, 2056, 89]


def _truncate(text, n=400):
    text = str(text)
    return text if len(text) <= n else text[:n] + f"... [{len(text) - n} more chars]"


def print_trace(result):
    print("=" * 100)
    print(f"task_id={result['task_id']}  task_type={result['task_type']}")
    info = result["agent_info"]

    if info.get("source") == "api":
        print(f"CRASHED: {info['content']}")
        print()
        return

    print(f"execution order: {' -> '.join(info['execution_order'])}")
    print(f"final subtask: {info['final_subtask_id']}")
    print("-" * 100)

    for step in result["traj"]:
        print(f"[{step['subtask_id']}] role={step['role']}  depends_on={step['depends_on'] or 'none'}")
        print(f"    description: {step['description']}")
        if step["context_received"]:
            print(f"    context received:\n{_indent(_truncate(step['context_received']), 8)}")
        else:
            print("    context received: (none)")
        n_calls = len(step["turns"])
        print(f"    llm/tool turns: {n_calls}  wall_time={step['wall_time_sec']}s  cost=${sum(c for c in step['usage']['cost'] if c is not None):.5f}")
        print(f"    output:\n{_indent(_truncate(step['output']), 8)}")
        print()

    print("-" * 100)
    print(f"FINAL ANSWER: {_truncate(info['end_reason']['content'])}")
    print(f"GROUND TRUTH: {result['gt_answer']}")
    print(f"parsed_answer: {info['end_reason']['parsed_answer']}")
    print(f"raw reward: {result['reward']}")

    score = normalize_score(result)
    passed = score == 1.0
    print(f"normalized score: {score:.3f}  {'PASS' if passed else 'FAIL'}")
    print(f"total_cost: ${info['total_cost']:.5f}   num_subtasks: {info['num_subtasks']}")
    print()


def _indent(text, spaces):
    pad = " " * spaces
    return "\n".join(pad + line for line in str(text).splitlines())


def main():
    results = []
    for task_id in TASK_IDS:
        print(f"\nRunning workflow architecture on task_id={task_id} ...")
        result = run_workflow(task_id, run_number=1)
        path = save_result(result, model=DEFAULT_MODEL)
        print(f"Saved to {path}")
        results.append(result)

    print("\n\n" + "#" * 100)
    print("# FULL EXECUTION TRACES")
    print("#" * 100)
    for result in results:
        print_trace(result)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for result in results:
        if result["agent_info"].get("source") == "api":
            print(f"task_id={result['task_id']:>5}  CRASHED  ({result['agent_info']['content']})")
            continue
        score = normalize_score(result)
        print(
            f"task_id={result['task_id']:>5}  task_type={result['task_type']:<32} "
            f"score={score:.3f}  {'PASS' if score == 1.0 else 'FAIL':4}  "
            f"cost=${result['agent_info']['total_cost']:.5f}"
        )


if __name__ == "__main__":
    main()
