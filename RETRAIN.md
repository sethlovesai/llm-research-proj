# Retraining the router on new tasks

Work on a new git branch of this repo, running commands from its root. Use Python 3.12 for new experiments; the refit alone needs only numpy.

## A. What each task needs
1. **One DAG** at `decomposition/cache/<id>.json` (copy an existing file's keys). The task list is that folder's contents. `category` (the CRMArena-Pro `task_type`) defines the leave-one-category-out folds. [verified by reading code]
2. **All three architectures, equal repeats.** Missing ones raise "missing data" (`router.py:144`); unequal repeats fail `router_dist.py:131`. [verified by running]
3. **One results file per run:** a JSON list of one dict (`task_id`, `run_number`, `task_type`, `gt_answer`, `reward`, `agent_info`, `traj`), named per `RESULT_GLOBS` (`oracle.py:110-114`):
   - `results/custom_baseline/results_gpt-4o-mini_react_task<id>_run<n>.json`
   - `results/workflow/results_gpt-4o-mini_workflow_task<id>_run<n>.json`
   - `results/multi_agent/results_gpt-4o-mini_multi_agent_task<id>_run<n>.json`

   [verified by running]
4. **Reward** (scored 0–1 by `normalize_score.py`): `0` or `1` for `exact_match` and `privacy_rejection`; for `fuzzy_match`, a dict whose `"f1"` is used. **Cost:** `agent_info.total_cost` (`oracle.py:203`, `router_dist.py:81`). Runs with `agent_info.source == "api"` (infrastructure errors) are skipped. [verified by reading code]

## B. Producing them (needs `.env`, see `.env.example`) [not run]
5. `pip install -r requirements.txt datasets`
6. Decompose each new task:
   ```
   python3 -c "import custom_tasks; from crm_sandbox.data.assets import TASKS_B2C; from decomposition.decompose import decompose_task; t=next(x for x in TASKS_B2C if x['idx']==<id>); decompose_task(t['idx'], t['query'], category=t['task'])"
   ```
7. Run the three architectures:
   ```
   python3 run_custom_tasks.py
   python3 architectures/run_workflow_batch.py
   python3 architectures/run_multi_agent_batch.py
   ```
   `run_workflow_batch.py` lacks `import custom_tasks`; add it for custom ids (900001+).

## C. Edits when the task set changes
8. Make these edits: [verified by running: 8a, 8b, 8f on a simulated task]

   a. `normalize_score.py:38-66` `TASK_TYPE_TO_REWARD_METRIC`: add any new `task_type`, or scoring raises `ValueError`.

   b. `router_dist.py:322` (`== 11`, scored) and `:324` (`== 13`, all-architecture quality ties): new counts (after 8a):
   `python3 -c "import router_dist as d; t=d.load('structured'); print(sum(x.scored for x in t), sum(not x.scored for x in t))"`

   c. `TASK_IDS` in `run_custom_tasks.py:37`, `architectures/run_workflow_batch.py:21` and `architectures/run_multi_agent_batch.py:34`.

   d. Different model: edit `oracle.py:111-113`, `run_custom_tasks.py:43`, `architectures/workflow.py:59` and `decomposition/decompose.py:18`, else new files are silently ignored.

   e. New custom tasks: add to `custom_tasks.py`. If F1 is unsuitable, add the id to `STRUCTURED_TASK_IDS` (`oracle.py:168`) and a scorer in `structured_score.py:345-349`.

   f. `task_tiers.py:19`: add a tier label (label only; fit unchanged).

   g. Optional: add new privacy tasks to `PRIV_TASKS` (`router_dist.py:40`) for sensitivity (f).

   **Commit before refitting.** After 8a, 8d or 8e (`normalize_score.py`, `oracle.py`, `structured_score.py`, `decomposition/*.py`), run `git add -A && git commit -m "..."` first, or the refit ends with `AssertionError` at `router_dist.py:488` after computing. 8b, 8c, 8f, new DAGs and results need none. [verified by running]

## D. Refit and check [verified by running]
9. Copy the report aside, refit, compare:
   ```
   cp results/router_dist/report.txt /tmp/report_old.txt
   python3 router_dist.py
   diff /tmp/report_old.txt results/router_dist/report.txt | less
   ```
   Outputs (`results/router_dist/`): `bucket_model.json`, `report.txt`, `summary_all.csv`, `per_task_headline_LOCO.csv`, `comparison_with_router_py.csv`.
10. Check: exit 0 ending `wrote: …`; new ids in the scored or excluded lists; expected `n` (categories) per width in the final bucket table; posterior means within 0–1. On unchanged data, `report.txt` is byte-identical (`cmp`).

## E. Update the prototype
11. Build the prototype data files: [verified by running]
    ```
    mkdir -p prototype/data
    python3 prototype/tools/make_evidence.py
    python3 prototype/tools/make_v1_examples.py
    ```
12. In the app repo (https://github.com/sethlovesai/llm-agent-architecture-router, branch `clean-submission`): [not run]
    - copy both new files and `results/router_dist/bucket_model.json` into `data/`;
    - hand-edit `data/SOURCE.md` (source, sha256s, date);
    - redeploy (HANDOVER.md §5).

## F. Known quirks
- The "protected files/dirs modified vs HEAD" line is a hard guard inside git (aborts the refit; section C) and meaningless outside git (always `none`). These guards come from the pre-registered analysis (`results/router_dist/SPEC.md`).
- `architectures/test_workflow.py` and `test_multi_agent.py` are live-API validation runs needing a key and the sandbox, not offline tests; `test_workflow.py` cites `decomposition/validate_decompose.py` (not in this repo). Offline check: `report.txt` byte-identical.
- Stale comments exist (e.g. `oracle.py:86`: single-agent did run on 900001/900002).
- `CLAUDE.md` is cited in comments, not shipped.
