"""
Reusable scoring-normalization layer for CRMArena-Pro result JSONs.

run_tasks.py / run_custom_tasks.py write one result dict per (task_id, run)
with a 'task_type' and a 'reward' field. Depending on the task's underlying
reward_metric (a property of the task category in the CRMArena-Pro dataset,
see crm_sandbox.data.assets.TASKS_B2C[i]["reward_metric"]), 'reward' is
shaped differently:

  - exact_match / privacy_rejection tasks: reward is already a 0/1 scalar.
  - fuzzy_match tasks (open-ended text answers): reward is a dict
    {"em": bool, "f1": float, "bleu": float, "rouge": {...}}.

normalize_score() collapses any of these into a single continuous quality
value in [0, 1], so scores are comparable across task categories and across
agent architectures (single-agent, workflow, multi-agent) that all reuse
this same result schema.

Usage:
    from normalize_score import normalize_score

    result = json.load(open(path))[0]   # one element of a results_*.json list
    score = normalize_score(result)     # float in [0, 1]

Run directly to summarize a directory of result files:
    python normalize_score.py [results_dir]
"""
import glob
import json
import sys
from collections import defaultdict

# task_type -> reward_metric, as defined by the CRMArena-Pro dataset
# (crmarena_pro["b2c"]["reward_metric"], one value per task category).
# Static property of the benchmark's task categories, not of any particular
# agent run, so this mapping holds regardless of which architecture produced
# the results being scored.
TASK_TYPE_TO_REWARD_METRIC = {
    "activity_priority": "exact_match",
    "best_region_identification": "exact_match",
    "case_routing": "exact_match",
    "confidential_company_knowledge": "privacy_rejection",
    "conversion_rate_comprehension": "exact_match",
    # Custom, author-designed tasks (not in the real CRMArena-Pro dataset) --
    # see custom_tasks.py. Both use fuzzy_match; see that module's docstring
    # for why exact_match doesn't fit either task's answer shape.
    "churn_risk_assessment": "fuzzy_match",
    "dual_signal_account_flagging": "fuzzy_match",
    "handle_time": "exact_match",
    "internal_operation_data": "privacy_rejection",
    "invalid_config": "exact_match",
    "knowledge_qa": "fuzzy_match",
    "lead_qualification": "exact_match",
    "lead_routing": "exact_match",
    "monthly_trend_analysis": "exact_match",
    "named_entity_disambiguation": "exact_match",
    "policy_violation_identification": "exact_match",
    "private_customer_information": "privacy_rejection",
    "quote_approval": "exact_match",
    "sales_amount_understanding": "exact_match",
    "sales_cycle_understanding": "exact_match",
    "sales_insight_mining": "fuzzy_match",
    "top_issue_identification": "exact_match",
    "transfer_count": "exact_match",
    "wrong_stage_rectification": "exact_match",
}


def get_reward_metric(task_type):
    try:
        return TASK_TYPE_TO_REWARD_METRIC[task_type]
    except KeyError:
        raise ValueError(
            f"Unknown task_type {task_type!r} -- add it to TASK_TYPE_TO_REWARD_METRIC "
            f"in normalize_score.py before scoring it."
        )


def normalize_score(result):
    """
    result: one element of a results_*.json list -- a dict with at least
    'task_type' and 'reward' keys, in the schema written by run_tasks.py /
    run_custom_tasks.py.

    Returns a single float in [0, 1]:
      - exact_match, privacy_rejection -> the existing 0/1 reward, unchanged.
      - fuzzy_match                    -> the reward dict's 'f1' score (0-1,
                                           continuous partial credit).
    """
    task_type = result["task_type"]
    reward = result["reward"]
    metric = get_reward_metric(task_type)

    if metric in ("exact_match", "privacy_rejection"):
        if isinstance(reward, dict):
            raise ValueError(
                f"task_type={task_type!r} has metric={metric!r} but reward is a dict "
                f"({reward!r}); expected a 0/1 scalar. Result schema may have changed."
            )
        return float(reward)

    if metric == "fuzzy_match":
        if not isinstance(reward, dict) or "f1" not in reward:
            raise ValueError(
                f"task_type={task_type!r} has metric='fuzzy_match' but reward "
                f"({reward!r}) is not a dict with an 'f1' key."
            )
        return float(reward["f1"])

    raise ValueError(f"Unhandled reward_metric {metric!r} for task_type {task_type!r}")


def summarize(results_dir):
    files = sorted(glob.glob(f"{results_dir}/*.json"))
    if not files:
        raise ValueError(f"No result files found in {results_dir}")

    by_cat = defaultdict(list)
    for f in files:
        result = json.load(open(f))[0]
        by_cat[result["task_type"]].append(normalize_score(result))

    print(f"{'category':35s} {'reward_metric':18s} {'avg_score':>10s} {'n':>5s}")
    print("-" * 72)
    all_scores = []
    for cat in sorted(by_cat):
        scores = by_cat[cat]
        all_scores.extend(scores)
        avg = sum(scores) / len(scores)
        print(f"{cat:35s} {get_reward_metric(cat):18s} {avg:10.1%} {len(scores):5d}")

    print("-" * 72)
    overall = sum(all_scores) / len(all_scores)
    print(f"{'OVERALL':35s} {'':18s} {overall:10.1%} {len(all_scores):5d}")
    return overall


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/custom_baseline"
    summarize(results_dir)
