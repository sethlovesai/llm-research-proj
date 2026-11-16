"""
Wrapper to run a specific, hand-picked list of task_ids (spanning multiple
categories) multiple times each, for baseline single-agent benchmarking.

run_tasks.py only supports "first N tasks in a category" (--num_tasks) or
"all tasks in category/categories" -- it has no way to select arbitrary
task_ids or repeat a task_id more than once (its checkpoint logic explicitly
skips task_ids it has already seen: run_tasks.py:92-94). This script
reimplements the minimal subset of run_tasks.py's ChatEnv/ChatAgent wiring
needed to do both, reusing the same classes so scoring stays identical.

NOTE: ChatAgent.act(env, index, temperature=...) previously ignored the
  `temperature` argument -- crm_sandbox/agents/chat_agent.py hardcoded
  temperature=0.0 in the real completion() call (chat_agent.py:130). That
  has been patched (chat_agent.py:130 now uses the passed-in `temperature`),
  so REPEAT_TEMPERATURE below actually takes effect for react-strategy runs.
  tool_call_agent.py's completion call is a separate hardcoded 0.0 and was
  NOT touched (not used by this script, which only runs react via ChatAgent).

Output: one JSON file per (task_id, run_number) under LOG_DIR, each
containing a single-element list with the same schema run_tasks.py writes
(task_id, task_type, gt_answer, reward, agent_info, traj), plus run_number.
"""
import json
import os
import traceback
from dotenv import load_dotenv

from crm_sandbox.data.assets import TASKS_B2C, B2C_SCHEMA, EXTERNAL_FACING_TASKS
from crm_sandbox.env.env import ChatEnv
from crm_sandbox.agents import ChatAgent
import custom_tasks  # noqa: F401 -- side effect: extends TASKS_B2C with custom task_ids 900001/900002

load_dotenv()

# ---- Fill in before running ----
TASK_IDS = [
    722, 1942, 672, 1857, 122, 125, 792, 1030, 1089, 2056, 1463, 1563,
    89, 36, 446, 421, 334, 348, 1936, 1914, 1624, 1654,
    900001, 900002,  # custom tasks (custom_tasks.py)
]
NUM_REPEATS = 3
MODEL = "gpt-4o-mini"
LLM_PROVIDER = "openai"
AGENT_STRATEGY = "react"
ORG_TYPE = "b2c"
AGENT_EVAL_MODE = "default"
MAX_TURNS = 20
PRIVACY_AWARE_PROMPT = False
REPEAT_TEMPERATURE = 0.7  # now actually honored by chat_agent.py (see module docstring)
LOG_DIR = "results/custom_baseline"
# ---------------------------------


def main():
    if not TASK_IDS:
        raise ValueError("TASK_IDS is empty -- paste your task_id list before running.")

    os.makedirs(LOG_DIR, exist_ok=True)

    tasks_by_id = {t["idx"]: t for t in TASKS_B2C if t["idx"] in TASK_IDS}
    missing = set(TASK_IDS) - set(tasks_by_id.keys())
    if missing:
        print(f"WARNING: {len(missing)} task_id(s) not found in TASKS_B2C: {sorted(missing)}")

    env = ChatEnv(tasks=tasks_by_id, org_type=ORG_TYPE)

    for task_id in TASK_IDS:
        if task_id not in tasks_by_id:
            continue
        task = tasks_by_id[task_id]
        agent_type = "external" if task["task"] in EXTERNAL_FACING_TASKS else "internal"

        for run_num in range(1, NUM_REPEATS + 1):
            out_path = f"{LOG_DIR}/results_{MODEL}_{AGENT_STRATEGY}_task{task_id}_run{run_num}.json"
            if os.path.exists(out_path):
                print(f"Skipping task_id={task_id} run={run_num} (output already exists: {out_path})")
                continue

            print(f"\n=== task_id={task_id} ({task['task']}) run {run_num}/{NUM_REPEATS} ===")
            agent = ChatAgent(
                model=MODEL,
                schema_obj=B2C_SCHEMA,
                eval_mode=AGENT_EVAL_MODE,
                max_turns=MAX_TURNS,
                strategy=AGENT_STRATEGY,
                provider=LLM_PROVIDER,
                interactive=False,
                agent_type=agent_type,
                privacy_aware_prompt=PRIVACY_AWARE_PROMPT,
            )
            try:
                reward = agent.act(env, task_id, temperature=REPEAT_TEMPERATURE)
                result = {
                    "task_id": task_id,
                    "run_number": run_num,
                    "task_type": task["task"],
                    "gt_answer": task["answer"],
                    "reward": reward,
                    "agent_info": agent.info,
                    "traj": agent.get_messages(),
                }
            except Exception as e:
                traceback.print_exc()
                result = {
                    "task_id": task_id,
                    "run_number": run_num,
                    "task_type": task["task"],
                    "gt_answer": task["answer"],
                    "reward": 0,
                    "agent_info": {"source": "api", "content": "Error: " + str(e)},
                    "traj": agent.get_messages(),
                }

            print("✅" if result["reward"] == 1 else "❌", f"task_id={task_id} run={run_num}")
            with open(out_path, "w") as f:
                json.dump([result], f, indent=2)


if __name__ == "__main__":
    main()
