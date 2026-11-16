# Handover

Author: Seth Wing Shen Yap

## 1. What this is
Streamlit prototype (`app_v2.py`), Master's AI capstone. An LLM turns a CRM task into a subtask DAG; a width-only Beta router recommends single-agent, workflow or multi-agent. In evaluation it did not beat the best fixed architecture.

## 2. Where things live
The app repo (https://github.com/sethlovesai/llm-agent-architecture-router, branch `clean-submission`) holds the app and frozen model.

This repo holds everything to refit: `router_dist.py`, `router.py`, `oracle.py`, `normalize_score.py`, `task_tiers.py`, `structured_score.py`, `decomposition/cache/` (24 DAGs) and `results/{custom_baseline,workflow,multi_agent}/` (216 run JSONs).

## 3. Key files (app repo)
| File | What it does | Touch it when |
|---|---|---|
| `data/bucket_model.json` | Frozen fitted model | Every refit |
| `data/bucket_evidence.json` | Task ids/categories per width | Every refit |
| `data/SOURCE.md` | Model provenance | Every refit |
| `router_adapter.py` | Loads model, recommends | Router logic changes |
| `app_v2.py` | Interface | Interface changes |
| `live_decompose.py` | Decomposer bridge, holds the pin | `decompose.py` changes |
| `research/decompose.py` | Verbatim research decomposer | Leave: pinned by sha256 |

## 4. Retrain on new data
Full procedure and edits: `RETRAIN.md`.
Work on a branch of this repo.
1. Refit: numpy only (tested: Python 3.12.10, numpy 2.3.0). New experiments: `pip install -r requirements.txt datasets`. [verified by running]
2. Name runs `results/custom_baseline/results_gpt-4o-mini_react_task{id}_run{n}.json` (`workflow/…_workflow_…`, `multi_agent/…_multi_agent_…`), per `oracle.py:110-113`. Misnamed files raise `missing data` (`router.py:144`). [verified by running]
3. Per new task, add `decomposition/cache/{id}.json` and all three architectures' results. `decompose_task()` (`decomposition/decompose.py:319`) creates it (needs LLM key). `TASK_IDS` in `architectures/run_workflow_batch.py:21` omits 900001/900002 and the script lacks `import custom_tasks`; add both to rerun custom tasks. [verified by reading code]
4. Add it to `TASK_TIERS` (`task_tiers.py:19`). Missing: tier "unknown", fit unchanged. [verified by running]
5. First add new `task_type`s to `TASK_TYPE_TO_REWARD_METRIC` (`normalize_score.py:38-66`), else `ValueError`. Then edit `router_dist.py:322` (`== 11` scored) and `:324` (`== 13` excluded). Excluded = all architectures tie on quality (`oracle.py:267`). Commit protected-file edits before refitting (RETRAIN.md section C). [verified by running]
6. Copy `report.txt` aside; run `python3 router_dist.py` (offline). [verified by running, network blocked]
7. Outputs: `results/router_dist/bucket_model.json`, `report.txt`, `summary_all.csv`, `per_task_headline_LOCO.csv`, `comparison_with_router_py.csv`. Check (unmodified data): `report.txt` byte-identical to shipped (`cmp`). `bucket_model.json` sha256 may differ by machine (float last digit, ~1e-15); reference (step-1 versions): `90066740b062f0fc9566ce3ee98d78fcc2a0cb4cb1b424890f038f5eb9215d38`. [verified by running]

## 5. Update the prototype after a refit
1. Copy the new `bucket_model.json` to `data/`.
2. Hand-edit `data/SOURCE.md` (source, commit, both sha256, copy time); no script writes it. [verified by reading code]
3. After `mkdir -p prototype/data`, this repo's `prototype/tools/make_evidence.py` and `make_v1_examples.py` write `bucket_evidence.json` and `v1_examples.json` there, offline from `decomposition/cache/`; copy both into the app repo's `data/`. `v2_recorded_examples.json`: demo-only, not regenerated. [verified by reading code]
4. `app_v2.py:160` and `:275` hard-code "24 tasks". [verified by reading code]
5. If `research/decompose.py` changes, update `PINNED_SHA256` and `PINNED_COMMIT` (`live_decompose.py:15-16`) or live decomposition is disabled. [verified by reading code]

## 6. Watch-outs
- Width-2 (2 categories) and width-4 (1) buckets are thin.
- 13/24 tasks scored zero everywhere; excluded from regret.
- Privacy scores unreliable: LLM judge sees refusal-check text.
- Licence: CC BY-NC 4.0, non-commercial only (`NOTICE.md`).
