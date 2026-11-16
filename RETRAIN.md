# Retraining the router on new tasks

Work in a copy of this folder, running commands from its root.

## A. What each task needs
1. **One DAG** at `decomposition/cache/<id>.json` (copy an existing file's keys). The task list is whatever is in this folder. `category` (the CRMArena-Pro `task_type`) defines the leave-one-category-out folds. [verified by reading code]
2. **Results for all three architectures, with the same number of repeats.** If any are missing, `router.py:144` raises "missing data". Unequal repeats fail the assert at `router_dist.py:131`. [verified by running]
3. **One results file per run:** a JSON list holding one dict with keys `task_id`, `run_number`, `task_type`, `gt_answer`, `reward`, `agent_info`, `traj`. Filenames follow `RESULT_GLOBS` (`oracle.py:110-114`):
   - `results/custom_baseline/results_gpt-4o-mini_react_task<id>_run<n>.json`
   - `results/workflow/results_gpt-4o-mini_workflow_task<id>_run<n>.json`
   - `results/multi_agent/results_gpt-4o-mini_multi_agent_task<id>_run<n>.json`

   [verified by running]
4. **Reward shapes** (converted to a 0–1 score by `normalize_score.py`):
   - `exact_match` and `privacy_rejection`: `0` or `1`
   - `fuzzy_match`: a dict containing `"f1"`, which is the value used

   **Cost** is read from `agent_info.total_cost` (`oracle.py:203`, `router_dist.py:81`). A run with `agent_info.source == "api"` counts as an infrastructure error and is skipped. [verified by reading code]

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

   b. `router_dist.py:322` (`== 11`, scored tasks) and `:324` (`== 13`, tasks where every architecture ties on quality): set the new counts, obtained (after 8a) with:
   `python3 -c "import router_dist as d; t=d.load('structured'); print(sum(x.scored for x in t), sum(not x.scored for x in t))"`

   c. `TASK_IDS` in `run_custom_tasks.py:37`, `architectures/run_workflow_batch.py:21` and `architectures/run_multi_agent_batch.py:34`.

   d. A different model: change `oracle.py:111-113`, `run_custom_tasks.py:43`, `architectures/workflow.py:59` and `decomposition/decompose.py:18`. Otherwise the new files are silently ignored.

   e. New custom tasks (not in the dataset): add them to `custom_tasks.py`. If F1 is unsuitable, add the id to `STRUCTURED_TASK_IDS` (`oracle.py:168`) and write a scorer in `structured_score.py:345-349`.

   f. `task_tiers.py:19`: add a tier label (label only; the fit is unchanged).

   g. Optional: new privacy tasks go in `PRIV_TASKS` (`router_dist.py:40`) for sensitivity (f).

## D. Refit and check [verified by running]
9. Copy the old report aside, refit, then compare:
   ```
   cp results/router_dist/report.txt /tmp/report_old.txt
   python3 router_dist.py
   diff /tmp/report_old.txt results/router_dist/report.txt | less
   ```
   The refit writes `bucket_model.json`, `report.txt`, `summary_all.csv`, `per_task_headline_LOCO.csv` and `comparison_with_router_py.csv` to `results/router_dist/`.
10. Check:
    - it exits 0 and ends with `wrote: …`;
    - the new ids appear in the scored or excluded lists;
    - the final bucket table shows the expected `n` (categories) per width;
    - all posterior means lie between 0 and 1.

    On unchanged data, `report.txt` should be byte-identical (`cmp`).

## E. Update the prototype
11. Build the prototype data files: [verified by running]
    ```
    mkdir -p prototype/data
    python3 prototype/tools/make_evidence.py
    python3 prototype/tools/make_v1_examples.py
    ```
12. In the prototype repo (branch `clean-submission`): [not run]
    - copy both new files and `results/router_dist/bucket_model.json` into `data/`;
    - hand-edit `data/SOURCE.md` (source, sha256s, date);
    - redeploy (HANDOVER.md §5).

## F. Known quirks
- The "protected files/dirs modified" line is meaningful only inside a git repository; elsewhere it always prints `none`. These guards come from the pre-registered analysis (`results/router_dist/SPEC.md`), so inside git your deliberate edits will be flagged, and that is expected.
- Some comments are stale (e.g. `oracle.py:86`: single-agent did run on 900001/900002).
- `CLAUDE.md` is cited in comments but not shipped.
