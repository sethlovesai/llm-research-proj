# Research repo: architecture router (CRMArena-Pro B2C)

## What this is
Everything needed to refit the bucket-based Beta router that recommends single-agent, workflow or multi-agent for a CRM task: refit code, 24 cached DAGs (`decomposition/cache/`), 216 per-run results (`results/`), the pre-registered spec and report (`results/router_dist/`) and the experiment pipeline. The app lives in a separate repo (below).

## Quick start: refit (offline, numpy only, no key or `.env`)
```
cp results/router_dist/report.txt /tmp/report_shipped.txt
python3 router_dist.py
cmp /tmp/report_shipped.txt results/router_dist/report.txt
```
On unchanged data `report.txt` is byte-identical: that is the check. [verified by running] For reference only, with Python 3.12.10 and numpy 2.3.0 `bucket_model.json` has sha256 `90066740b062f0fc9566ce3ee98d78fcc2a0cb4cb1b424890f038f5eb9215d38`; elsewhere some floats can differ in the last digit and the hash will not match.

## Python version
Use Python 3.12 for new experiments: several pinned packages (numpy 2.0.1, pandas 2.2.2, lxml 5.2.2, jiter 0.5.0) have no prebuilt wheels on 3.13, so installs may fail; a source build might still work. [checked with pip download --only-binary on 3.13; not a full install] The refit alone needs only numpy (any recent version). [verified by running]

## New experiments [not run]
`pip install -r requirements.txt datasets`, an LLM API key and your own CRMArena-Pro Salesforce sandbox login in `.env` (variable names in `.env.example`). See RETRAIN.md.

## Predict on new tasks
Needs the app repo: `git clone -b clean-submission https://github.com/sethlovesai/llm-agent-architecture-router` (installs: see its `requirements.txt`). From its root:
```python
import live_decompose as ld, router_adapter as ra
model = ra.FrozenModel()   # or ra.FrozenModel("path/to/refit/results/router_dist/bucket_model.json")
g = ld.generate(text)      # LLM call, needs OPENAI_API_KEY; g.width
rec = ra.recommend(model, g.width, lam=100)   # lam in {0, 10, 100, 300}; rec["pick"], rec["P_best"], rec["no_clear_winner"], rec["flag"]
```
`FrozenModel` and `recommend` [verified by running]; `ld.generate` [not run].

## Where to start
HANDOVER.md, then RETRAIN.md.

## Licence
CC BY-NC 4.0, non-commercial use only; see NOTICE.md and LICENSE.txt.
