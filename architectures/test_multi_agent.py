"""
Validation run for architectures/multi_agent.py.

Two phases, per the validation plan:
1. THOROUGH validation on task_ids with real parallelism (89, 36 --
   lead_qualification, parallelism_width=4) -- prints the full execution
   trace (levels, per-subtask state/attempts, supervisor synthesis) so
   concurrent scheduling and failure handling can be checked by hand.
2. SANITY CHECK on width=1 task_ids (722, 2056) -- these have no real
   parallelism to exploit, so per-subtask outputs and execution structure
   should match workflow.py's run. NOTE: the supervisor synthesis step (see
   multi_agent.py's module docstring) is an extra LLM call workflow.py never
   makes, so the final answer text is NOT guaranteed byte-identical -- this
   script reports the workflow.py comparison explicitly rather than
   asserting equality, so you can see exactly where (if anywhere) they
   diverge and why.

Always uses normalize_score.py to report pass/fail (same reasoning as
test_workflow.py: don't trust a raw `reward == 1` check for fuzzy_match
tasks).

Run with: python architectures/test_multi_agent.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architectures.multi_agent import run_multi_agent, save_result, DEFAULT_MODEL, DEFAULT_LOG_DIR
from normalize_score import normalize_score

PARALLEL_TASK_IDS = [89, 36]
SANITY_TASK_IDS = [722, 2056]

WORKFLOW_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "workflow")


def _truncate(text, n=400):
    text = str(text)
    return text if len(text) <= n else text[:n] + f"... [{len(text) - n} more chars]"


def _indent(text, spaces):
    pad = " " * spaces
    return "\n".join(pad + line for line in str(text).splitlines())


def print_trace(result):
    print("=" * 100)
    print(f"task_id={result['task_id']}  task_type={result['task_type']}")
    info = result["agent_info"]

    if info.get("source") == "api":
        print(f"CRASHED: {info['content']}")
        print()
        return

    print("execution levels:")
    for i, level in enumerate(info["execution_levels"]):
        tag = " (concurrent)" if len(level) > 1 else ""
        print(f"  level {i}: {level}{tag}")
    print(f"final subtask: {info['final_subtask_id']}")
    if info["supervisor_synthesis"] is not None:
        print(f"supervisor verdict: {info['supervisor_synthesis']['verdict']}")
    else:
        print("supervisor verdict: (skipped -- trivial single-subtask task, nothing to synthesize)")
    print("-" * 100)

    for step in result["traj"]:
        state_note = f"  state={step['state']}" if step.get("state") else ""
        retry_note = f"  attempts={step['attempts']}{' FAILED_AFTER_RETRY' if step['failed_after_retry'] else ''}"
        print(f"[{step['subtask_id']}] role={step['role']}{state_note}{retry_note}")
        print(f"    description: {step['description']}")
        n_calls = len(step["turns"])
        cost = sum(c for c in step["usage"]["cost"] if c is not None)
        print(f"    llm/tool turns: {n_calls}  wall_time={step['wall_time_sec']}s  cost=${cost:.5f}")
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


def _load_workflow_result(task_id, run_number=1):
    path = os.path.join(WORKFLOW_LOG_DIR, f"results_{DEFAULT_MODEL}_workflow_task{task_id}_run{run_number}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)[0]


def compare_to_workflow(ma_result, wf_result):
    print(f"--- width=1 sanity check: task_id={ma_result['task_id']} vs. workflow.py run 1 ---")
    if wf_result is None:
        print("  (no workflow.py result found to compare against)")
        return
    if ma_result["agent_info"].get("source") == "api" or wf_result["agent_info"].get("source") == "api":
        print("  (one side crashed -- skipping score/output comparison)")
        return

    ma_score = normalize_score(ma_result)
    wf_score = normalize_score(wf_result)
    print(f"  workflow score={wf_score:.3f}   multi_agent score={ma_score:.3f}   "
          f"{'MATCH' if ma_score == wf_score else 'DIFFERS'}")

    ma_subtask_outputs = {t["subtask_id"]: t["output"] for t in ma_result["traj"] if t["role"] != "Supervisor"}
    wf_subtask_outputs = {t["subtask_id"]: t["output"] for t in wf_result["traj"]}
    same_subtasks = ma_subtask_outputs == wf_subtask_outputs
    print(f"  per-subtask outputs identical (excluding the new Supervisor step): {same_subtasks}")
    if not same_subtasks:
        for sid in sorted(set(ma_subtask_outputs) | set(wf_subtask_outputs), key=lambda x: (len(x), x)):
            wo = wf_subtask_outputs.get(sid)
            mo = ma_subtask_outputs.get(sid)
            if wo != mo:
                print(f"    [{sid}] workflow:    {_truncate(wo, 150)}")
                print(f"    [{sid}] multi_agent: {_truncate(mo, 150)}")

    final_match = ma_result["agent_info"]["end_reason"]["content"] == wf_result["agent_info"]["end_reason"]["content"]
    print(f"  final answer text identical: {final_match}  "
          f"(expected to sometimes differ -- see multi_agent.py's supervisor-synthesis caveat)")
    print()


def main():
    print("#" * 100)
    print("# PHASE 1: thorough validation on real-parallelism tasks (89, 36)")
    print("#" * 100)
    parallel_results = []
    for task_id in PARALLEL_TASK_IDS:
        print(f"\nRunning multi-agent architecture on task_id={task_id} ...")
        result = run_multi_agent(task_id, run_number=1)
        path = save_result(result, model=DEFAULT_MODEL)
        print(f"Saved to {path}")
        parallel_results.append(result)

    for result in parallel_results:
        print_trace(result)

    print("#" * 100)
    print("# PHASE 2: width=1 sanity check (722, 2056) vs. workflow.py")
    print("#" * 100)
    sanity_results = []
    for task_id in SANITY_TASK_IDS:
        print(f"\nRunning multi-agent architecture on task_id={task_id} ...")
        result = run_multi_agent(task_id, run_number=1)
        path = save_result(result, model=DEFAULT_MODEL)
        print(f"Saved to {path}")
        sanity_results.append(result)

    for result in sanity_results:
        print_trace(result)

    print("=" * 100)
    print("WORKFLOW COMPARISON (width=1 sanity check)")
    print("=" * 100)
    for result in sanity_results:
        wf_result = _load_workflow_result(result["task_id"])
        compare_to_workflow(result, wf_result)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for result in parallel_results + sanity_results:
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
