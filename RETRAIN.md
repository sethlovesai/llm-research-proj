# Retraining the router on new tasks

The refit groups tasks by DAG width (how many subtasks can run in parallel) and learns each architecture's quality and cost per width group.
Each new task needs a category (`task_type`), a description (`query`) and a verified correct answer, and must be answerable in the CRMArena-Pro Salesforce sandbox.
Without sandbox access you can only refit from result files in the Reference 1 format. Run commands from the repo root.

## Steps

### 0. Set up [not run]
```
git clone https://github.com/sethlovesai/llm-research-proj.git
cd llm-research-proj
git checkout -b my-new-tasks
python3.12 -m venv ../router-venv
source ../router-venv/bin/activate
pip install -r requirements.txt datasets
cp .env.example .env
```
Fill in `OPENAI_API_KEY` and the three `SALESFORCE_B2C_*` lines in `.env`. Keep the venv outside the repo: `.gitignore` does not exclude it and step 5c commits everything. Use Python 3.12: some pinned packages lack 3.13 wheels. The refit alone needs only numpy.

### 1. Add the tasks [verified by reading code]
- CRMArena-Pro dataset tasks: you need only their `idx` numbers.
- Your own tasks: add them to `CUSTOM_TASKS` in `custom_tasks.py`, copying task 900001's format: `idx` (900003 onwards), `task`, `persona`, `query`, `metadata`, `reward_metric`, `answer`. Verify answers in the sandbox first.

### 2. Decompose each task [not run]
Once per id; it writes `decomposition/cache/<id>.json` and skips cached ids.
```
python3 -c "import custom_tasks; from crm_sandbox.data.assets import TASKS_B2C; from decomposition.decompose import decompose_task; t=next(x for x in TASKS_B2C if x['idx']==<id>); decompose_task(t['idx'], t['query'], category=t['task'])"
```
The task list is whatever is in `decomposition/cache/`, one DAG per task.

### 3. Add the ids to the runners [verified by reading code]
Add the new ids to the end of `TASK_IDS` in `run_custom_tasks.py:37`, `architectures/run_workflow_batch.py:21` and `architectures/run_multi_agent_batch.py:34`. Keep the old ids: all three runners skip existing result files. For custom ids (900001+), also add `import custom_tasks` to `run_workflow_batch.py` (after its `sys.path.insert` line).

### 4. Run the three architectures [not run]
```
python3 run_custom_tasks.py
python3 architectures/run_workflow_batch.py
python3 architectures/run_multi_agent_batch.py
```
Each task runs 3 times per architecture: 5 tasks means 5 × 3 × 3 = 45 correctly named runs. Check each new task has 3 files per folder and none crashed (your ids first):
```
for id in 900003 900004; do
  for d in custom_baseline workflow multi_agent; do
    n=$(ls results/$d/*_task${id}_run*.json 2>/dev/null | wc -l)
    bad=$(grep -lE '"source": ?"api"' results/$d/*_task${id}_run*.json 2>/dev/null | wc -l)
    echo "$id $d files=$n crashed=$bad"
  done
done
```
A crashed run is saved with `agent_info.source == "api"`. Workflow and multi-agent runners print `CRASHED`; single-agent prints a traceback. [verified by reading code] The refit drops crashed runs from the averages, leaving that architecture short of repeats, so it stops with `AssertionError` at `router_dist.py:131`. [verified by running] Rerun crashed runs before refitting: `run_multi_agent_batch.py` reruns them itself; for the other two, delete the crashed file first.

### 5. Prepare the refit
a. Add any new `task_type` to `TASK_TYPE_TO_REWARD_METRIC` in `normalize_score.py:38-66`, or scoring raises `ValueError`. [verified by running]

b. Optional: add a tier label in `task_tiers.py:19` (label only; fit unchanged). [verified by running]

c. Commit everything: `git add -A && git commit -m "Add new tasks and results"`. The refit asserts at `router_dist.py:488` that `router.py`, `oracle.py`, `structured_score.py`, `normalize_score.py` and tracked files in `decomposition/` match the last commit, else it ends in `AssertionError` after computing. [verified by running]

d. Get the new counts (only after 5a):
```
python3 -c "import router_dist as d; t=d.load('structured'); print(sum(x.scored for x in t), sum(not x.scored for x in t))"
```
Put the first number into `router_dist.py:322` (now 11: scored tasks) and the second into `:324` (now 13: all architectures score the same). `router_dist.py` is unguarded, so no commit is needed. [verified by running]

### 6. Refit [verified by running]
```
cp results/router_dist/report.txt /tmp/report_old.txt
python3 router_dist.py
```
It writes `bucket_model.json`, `report.txt`, `summary_all.csv`, `per_task_headline_LOCO.csv` and `comparison_with_router_py.csv` to `results/router_dist/`.

### 7. Read the result [verified by running]
- Exit 0, ending `wrote: …`.
- Your new ids appear in the "Scored" or "Regret-excluded" list near the top of `report.txt`. Excluded tasks (all architectures score the same) still count in the fit, just not in regret.
- The final bucket table shows the expected `n` (categories) per width; posterior means lie in 0–1.
- Diff against the old report: `diff /tmp/report_old.txt results/router_dist/report.txt | less`.
- On unchanged data, `report.txt` is byte-identical (`cmp`).

Five new tasks may change little: each category is one evidence unit per width group.

### 8. Optional: update the app
Build the app data here: [verified by running]
```
mkdir -p prototype/data
python3 prototype/tools/make_evidence.py
python3 prototype/tools/make_v1_examples.py
```
Then, in the app repo (https://github.com/sethlovesai/llm-agent-architecture-router, branch `clean-submission`): [not run]
- copy both new files and `results/router_dist/bucket_model.json` into `data/`;
- hand-edit `data/SOURCE.md` (source, sha256s, date);
- redeploy (HANDOVER.md section 4).

## Reference
1. **Result and DAG files.** One file per run: a JSON list of one dict (`task_id`, `run_number`, `task_type`, `gt_answer`, `reward`, `agent_info`, `traj`), named per `RESULT_GLOBS` (`oracle.py:110-114`):
   - `results/custom_baseline/results_gpt-4o-mini_react_task<id>_run<n>.json`
   - `results/workflow/results_gpt-4o-mini_workflow_task<id>_run<n>.json`
   - `results/multi_agent/results_gpt-4o-mini_multi_agent_task<id>_run<n>.json`

   Reward (scored 0–1 by `normalize_score.py`): `0` or `1` for `exact_match` and `privacy_rejection`; for `fuzzy_match`, a dict whose `"f1"` is used. Cost: `agent_info.total_cost` (`oracle.py:203`, `router_dist.py:81`). `agent_info.source == "api"` marks an infrastructure error (step 4). A missing architecture raises "missing data" (`router.py:144`); unequal repeats fail `router_dist.py:131`. DAG files share the existing files' keys; `category` defines the leave-one-category-out folds. [verified by running]
2. **Changing the model.** Edit `oracle.py:111-113`, `run_custom_tasks.py:43`, `architectures/workflow.py:59` and `decomposition/decompose.py:18`, else new results are silently ignored. Commit before refitting (step 5c). [verified by reading code]
3. **Custom tasks needing a structured scorer.** If F1 is unsuitable, add the id to `STRUCTURED_TASK_IDS` (`oracle.py:168`) and a scorer in `structured_score.py:345-349`. [verified by reading code]
4. **Privacy tasks.** Optionally add new privacy tasks to `PRIV_TASKS` (`router_dist.py:40`) for sensitivity analysis (f). [verified by reading code]
5. **Known quirks.**
   - The "protected files/dirs modified vs HEAD" guard (files listed in step 5c) aborts the refit inside git; outside git it always prints `none`. It comes from the pre-registered analysis (`results/router_dist/SPEC.md`).
   - `architectures/test_workflow.py` and `test_multi_agent.py` are live-API runs (key and sandbox), not offline tests; `test_workflow.py` cites `decomposition/validate_decompose.py`, not in this repo.
   - Stale comments (e.g. `oracle.py:86`: single-agent did run on 900001/900002).
   - `CLAUDE.md` is cited but not shipped.
