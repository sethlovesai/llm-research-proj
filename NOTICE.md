# NOTICE

This kit is derived from Salesforce AI Research's CRMArena-Pro evaluation code
(https://github.com/SalesforceAIResearch/CRMArena), released under the Creative Commons
Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0). The full licence text is in `LICENSE.txt`.

It was modified by Seth Wing Shen Yap for a UTS Master of AI capstone project. Non-commercial use only.

## What is Salesforce's and what is not
Checked against the original repository's `origin/main`:

- Unmodified Salesforce files: `LICENSE.txt` and everything under `crm_sandbox/` except `agents/chat_agent.py`.
- Salesforce files modified for this project: `crm_sandbox/agents/chat_agent.py` (1 line, sampling temperature),
  `run_tasks.py`, `requirements.txt`.
- Added for this project: `router_dist.py`, `router.py`, `oracle.py`, `normalize_score.py`, `task_tiers.py`,
  `structured_score.py`, `custom_tasks.py`, `run_custom_tasks.py`, `architectures/`, `decomposition/`,
  `prototype/tools/`, everything under `results/`, `README.md`, `HANDOVER.md`, `RETRAIN.md` and `.env.example`.

## Not checked
- The licences of the Hugging Face datasets the code downloads (`Salesforce/CRMArenaPro`, `Salesforce/CRMArena`)
  were not checked.
- The files under `results/` contain text returned from the CRMArena-Pro Salesforce sandbox (synthetic data). Its
  data licence was not checked, so treat it as non-commercial as well.
- Third-party Python package licences were not checked.
