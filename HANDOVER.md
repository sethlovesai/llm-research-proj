# Handover

Author: Seth Wing Shen Yap

## 1. What this is
Research repo for the CRMArena-Pro B2C architecture router: refit code, 24 cached DAGs, 216 per-run results, pre-registered spec and report. In evaluation it did not beat the best fixed architecture.
The Streamlit app is separate: https://github.com/sethlovesai/llm-agent-architecture-router (branch `clean-submission`).

## 2. Key files [verified by reading code]
(P) = commit edits before refitting.

| File | Does | Touch when |
|---|---|---|
| `router_dist.py` | Width-only Beta router per `SPEC.md`; writes `results/router_dist/` | Task counts change |
| `router.py` | Earlier router; loaders reused by `router_dist.py` | Rarely (P) |
| `oracle.py` | Best architecture per task; result filename patterns | New model or structured-scored task (P) |
| `normalize_score.py` | Reward to 0–1 score per `task_type` | New `task_type` (P) |
| `task_tiers.py` | Tier labels per task | New task |
| `custom_tasks.py`, `structured_score.py` (P) | Define custom tasks (900001+); score 900001/900002 | New custom task |
| `decomposition/decompose.py`, `decomposition/cache/` | LLM task-to-DAG decomposition, cached per task | New task; (P) only for edits to `decompose.py` or existing cache files |
| `run_custom_tasks.py`, `architectures/run_workflow_batch.py`, `run_multi_agent_batch.py` | Single-agent, workflow, multi-agent runners; need `.env` (names in `.env.example`) | New task |
| `architectures/workflow.py`, `multi_agent.py` | Run a DAG sequentially, or in concurrent supervised waves | Architecture changes |
| `prototype/tools/make_evidence.py`, `make_v1_examples.py` | Build app data files offline | Every refit |
| `results/{custom_baseline,workflow,multi_agent}/`, `results/router_dist/` | One JSON per run; `SPEC.md` (pre-registration; do not edit), `report.txt` | New runs |

`crm_sandbox/` and `run_tasks.py` are Salesforce's (`NOTICE.md`).

## 3. Retrain on new data [verified by running]
Detail: `RETRAIN.md`.
1. Python 3.12 for new experiments; refit needs only numpy.
2. Per new task: one DAG and equal repeats of all three architectures.
3. Add new `task_type`s to `normalize_score.py` before the `router_dist.py:322`/`:324` counts.
4. Commit (P) edits first, else the `router_dist.py:488` assert aborts.
5. Check: `report.txt` byte-identical on unchanged data (`cmp`). `bucket_model.json` sha256 may differ (last float digit) elsewhere; reference (Python 3.12.10, numpy 2.3.0): `90066740b062f0fc9566ce3ee98d78fcc2a0cb4cb1b424890f038f5eb9215d38`.

## 4. Update the app (steps 2–7: app repo)
1. Here: `mkdir -p prototype/data`; run both builders. [verified by running]
2. Copy `results/router_dist/bucket_model.json` and both builder outputs to the app's `data/`. [not run]
3. Hand-edit `data/SOURCE.md` (source, commit, both sha256, copy time); no script writes it. [verified by reading code]
4. Update "24 tasks" at `app_v2.py:160` and `:275`. [verified by reading code]
5. If `research/decompose.py` changes, update the pin (`live_decompose.py:15-16`), else live decomposition is disabled. [verified by reading code]
6. `data/v2_recorded_examples.json` (recorded demo DAGs, no model output) needs no regeneration after a refit. [verified by reading code]
7. Redeploy per the app README. [not run]

## 5. Watch-outs
- Thin buckets: width 2 has 2 categories, width 4 has 1. [verified by running]
- 13 of 24 tasks scored zero for every architecture; excluded from regret. [verified by running]
- Privacy scores unreliable: the judge scores the trajectory, including refusal-check text (`crm_sandbox/env/env.py:562`). [verified by reading code]
- Licence: CC BY-NC 4.0, non-commercial only (`NOTICE.md`). [verified by reading code]
