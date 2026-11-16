"""
Workflow architecture: executes a task's cached decomposition
(decomposition/cache/{task_id}.json) as a fixed sequential pipeline.

Per CLAUDE.md's design decisions:
- Decomposition is computed once (by decomposition/decompose.py) and only
  read here, never regenerated -- this module isolates "how well does an
  architecture execute a given DAG," not "how differently did it plan."
- Execution order is a pure, deterministic topological sort of the DAG
  (Kahn's algorithm, ties broken by ascending numeric subtask id). There is
  no LLM call and no agent-level reasoning involved in ordering -- that is
  what distinguishes "workflow" from "multi-agent supervisor".
- Three fixed roles: Retriever (CRM tool access, no reasoning), Analyst
  (reasoning over already-retrieved context, no tools), Responder (formats
  the final answer, no tools).
- Retriever subtasks reuse this repo's existing single-agent tool-calling
  setup: the <thought>/<execute>/<respond> ReAct loop against a live
  Salesforce sandbox via SalesforceConnector.run_query, the same schema/
  prompt templates from crm_sandbox.agents.prompts, and
  ChatAgent.message_action_parser for parsing -- the same mechanism
  run_custom_tasks.py's single-agent baseline uses (NOT functions.py's
  ToolEnv/tool_call_agent.py path, which CLAUDE.md documents as hard-
  restricted to org_type=="original" and unused in this project).
- Scoring reuses crm_sandbox.env.env.Evaluator (the exact class ChatEnv
  uses for the single-agent baseline) and normalize_score.py, so results
  are directly comparable across architectures. The evaluator's own LLM
  calls (e.g. parse_answers' extraction fallback) are scoring overhead and
  are intentionally excluded from this module's own cost accounting.
- Output schema mirrors results/custom_baseline/ (task_id, run_number,
  task_type, gt_answer, reward, agent_info, traj) so results are directly
  comparable to the single-agent baseline; traj here is a per-subtask
  execution trace rather than a flat message list.
"""
import json
import os
import sys
import time
import traceback

from dotenv import load_dotenv
from litellm import completion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crm_sandbox.agents.chat_agent import ChatAgent
from crm_sandbox.agents.prompts import (
    REACT_EXTERNAL_PROMPT,
    REACT_INTERNAL_PROMPT,
    REACT_RULE_STRING,
    SCHEMA_STRING,
    SYSTEM_METADATA,
)
from crm_sandbox.data.assets import B2C_SCHEMA, EXTERNAL_FACING_TASKS, TASKS_B2C
from crm_sandbox.env.connect_sandbox import SalesforceConnector
from crm_sandbox.env.env import Evaluator

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
# Matches run_custom_tasks.py's REPEAT_TEMPERATURE -- CLAUDE.md requires
# model/temperature be held fixed and identical across all three
# architectures, so architecture is the only variable under test.
DEFAULT_TEMPERATURE = 0.7
# Same evaluator model/provider ChatEnv uses by default for the single-agent
# baseline (crm_sandbox/env/env.py ChatEnv.__init__ defaults) -- kept
# identical here so scores are comparable across architectures.
EVALUATOR_MODEL = "gpt-4o-2024-08-06"
EVALUATOR_PROVIDER = "openai"
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "decomposition", "cache")
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "workflow")
MAX_RETRIEVER_TURNS = 8

ANALYST_SYSTEM_PROMPT = """\
You are an Analyst in a CRM task pipeline. You reason, compute, or judge over \
data that other subtasks have already retrieved -- you have no tools and no \
direct access to the CRM system, only the context given to you below.

# Overall task
{task_query}

# Your subtask
{subtask_description}

{context_block}

Carry out your subtask using only the context above. \
Your response must be an explicit judgment. \
State clearly whether the evidence satisfies what you were asked to check (e.g. 'Yes, satisfied: [brief reason]' or 'No, not satisfied: [brief reason]'), \ 
not just a restatement of what you were asked to check. \
Respond with only the \
result of your analysis, precise and concise, in a form usable by whatever \
subtask consumes it next. Do not add explanation beyond what your subtask asks for.
If the context shows no data was found, or the retrieval failed, state that explicitly rather than guessing \ 
, this is a valid, informative answer, not a failure to complete your subtask."""

RESPONDER_SYSTEM_PROMPT = """\
You are the Responder in a CRM task pipeline -- the final step. You have no \
tools and no direct access to the CRM system; you only have the context below.

# Overall task (this is the query the final answer must satisfy)
{task_query}

# Your subtask
{subtask_description}

{context_block}

Using only the context above, produce the final answer to the overall task. \
Respond with ONLY the final answer, formatted exactly as the overall task \
requests (e.g. an Id, a bare list of values, a short phrase) -- no explanation, \
no preamble, no extra text."""

RETRIEVER_TASK_TEMPLATE = """\
# Overall task
{task_query}

# Your subtask
{subtask_description}

{context_block}

IMPORTANT : Return ONLY the raw data you found (exact field values, notes text, records) \
Do not interpret, judge, or summarize what it means. \ 
Judgment happens in a later step, not here. This is NOT necessarily the \
final answer to the overall task. \ 
Once you have the data this subtask asks \
for, submit it with <respond> exactly as retrieved (e.g. raw field values or a \
comma-separated list of records), without additional formatting or explanation."""


class WorkflowError(Exception):
    """Raised for workflow-execution failures not caused by an underlying LLM/tool error."""


def _load_decomposition(task_id, cache_dir):
    path = os.path.join(cache_dir, f"{task_id}.json")
    if not os.path.exists(path):
        raise WorkflowError(
            f"No cached decomposition found for task_id={task_id} at {path}. "
            "Run decomposition/decompose.py for this task first."
        )
    with open(path) as f:
        return json.load(f)


def _load_task(task_id):
    for t in TASKS_B2C:
        if t["idx"] == task_id:
            return t
    raise WorkflowError(f"task_id={task_id} not found in TASKS_B2C.")


def _build_schema(schema_obj):
    object_description = {}
    for item in schema_obj:
        object_description[item["object"]] = "\n".join(f"  - {k}: {v}" for k, v in item["fields"].items())
    return SCHEMA_STRING.format(
        object_names=", ".join(object_description.keys()),
        object_fields="\n".join(f"{obj}\n{fields}" for obj, fields in object_description.items()),
    )


def _topological_order(subtasks):
    """Deterministic execution order derived purely from the DAG structure
    (Kahn's algorithm; ties broken by ascending numeric subtask id). No
    agent-level reasoning about ordering is involved -- see module docstring."""
    subtasks_by_id = {s["id"]: s for s in subtasks}
    indegree = {sid: 0 for sid in subtasks_by_id}
    children = {sid: [] for sid in subtasks_by_id}
    for s in subtasks:
        for dep in s["depends_on"]:
            if dep not in subtasks_by_id:
                raise WorkflowError(f"Subtask {s['id']} depends on unknown id '{dep}'")
            children[dep].append(s["id"])
            indegree[s["id"]] += 1

    def sort_key(sid):
        try:
            return (0, int(sid))
        except ValueError:
            return (1, sid)

    ready = sorted([sid for sid, d in indegree.items() if d == 0], key=sort_key)
    order = []
    while ready:
        ready.sort(key=sort_key)
        current = ready.pop(0)
        order.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if len(order) != len(subtasks):
        raise WorkflowError("Cycle detected in subtask DAG -- cannot compute a fixed execution order.")
    return order


def _build_context_block(subtask, outputs_by_id, subtasks_by_id):
    deps = subtask["depends_on"]
    if not deps:
        return ""
    parts = ["# Context from prior subtasks"]
    for dep_id in deps:
        dep = subtasks_by_id[dep_id]
        parts.append(
            f"--- Output of Subtask {dep_id} ({dep['role']}): {dep['description']} ---\n"
            f"{outputs_by_id[dep_id]}"
        )
    return "\n\n".join(parts)


def _new_usage():
    return {"cost": [], "completion_tokens": [], "prompt_tokens": [], "total_tokens": []}


def _record_usage(usage, res):
    for key in ("completion_tokens", "prompt_tokens", "total_tokens"):
        usage[key].append(getattr(res.usage, key, 0))
    usage["cost"].append(res._hidden_params.get("response_cost"))


def _merge_usage(total, part):
    for key in total:
        total[key].extend(part[key])


def _run_retriever_subtask(subtask, task_query, context_block, task_metadata, agent_type,
                            schema_str, sf_connector, model, temperature, max_turns=MAX_RETRIEVER_TURNS):
    base_prompt = REACT_EXTERNAL_PROMPT if agent_type == "external" else REACT_INTERNAL_PROMPT
    sys_prompt = base_prompt.format(system_description=schema_str, system="Salesforce instance")
    if task_metadata and task_metadata.get("required"):
        sys_prompt += SYSTEM_METADATA.format(system_metadata=task_metadata["required"], system="Salesforce instance")

    user_prompt = RETRIEVER_TASK_TEMPLATE.format(
        task_query=task_query,
        subtask_description=subtask["description"],
        context_block=context_block,
    )
    messages = [
        {"role": "system", "content": sys_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    usage = _new_usage()
    turns = []
    output = None
    for turn_num in range(1, max_turns + 1):
        res = completion(messages=messages, model=model, temperature=temperature, max_tokens=2000, top_p=1.0)
        _record_usage(usage, res)
        message = res.choices[0].message.model_dump()
        content = message["content"].strip()
        action = ChatAgent.message_action_parser(message, model)
        messages.append({"role": "assistant", "content": content})

        if action is None:
            messages.append({"role": "user", "content": REACT_RULE_STRING})
            turns.append({"turn": turn_num, "assistant": content, "action": None, "observation": None})
            continue

        if action["name"] == "execute":
            observation, _status = sf_connector.run_query(action["content"])
            observation = str(observation)
            messages.append({"role": "user", "content": f"Salesforce instance output: {observation}"})
            turns.append({"turn": turn_num, "assistant": content, "action": action, "observation": observation})
        elif action["name"] == "respond":
            turns.append({"turn": turn_num, "assistant": content, "action": action, "observation": None})
            output = action["content"]
            break

    if output is None:
        output = f"Error: Retriever subtask {subtask['id']} did not produce a <respond> within {max_turns} turns."

    return output, turns, usage


def _run_analyst_subtask(subtask, task_query, context_block, model, temperature):
    prompt = ANALYST_SYSTEM_PROMPT.format(
        task_query=task_query,
        subtask_description=subtask["description"],
        context_block=context_block,
    ).strip()
    messages = [{"role": "user", "content": prompt}]
    usage = _new_usage()
    res = completion(messages=messages, model=model, temperature=temperature, max_tokens=1500, top_p=1.0)
    _record_usage(usage, res)
    output = res.choices[0].message.content.strip()
    return output, [{"prompt": prompt, "response": output}], usage


def _run_responder_subtask(subtask, task_query, context_block, model, temperature):
    prompt = RESPONDER_SYSTEM_PROMPT.format(
        task_query=task_query,
        subtask_description=subtask["description"],
        context_block=context_block,
    ).strip()
    messages = [{"role": "user", "content": prompt}]
    usage = _new_usage()
    res = completion(messages=messages, model=model, temperature=temperature, max_tokens=1500, top_p=1.0)
    _record_usage(usage, res)
    output = res.choices[0].message.content.strip()
    return output, [{"prompt": prompt, "response": output}], usage


def run_workflow(task_id, run_number=1, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE,
                  cache_dir=DEFAULT_CACHE_DIR, max_retriever_turns=MAX_RETRIEVER_TURNS, org_type="b2c"):
    """Executes task_id's cached decomposition as a fixed sequential pipeline
    and scores the final subtask's output. Returns a result dict in the same
    schema as results/custom_baseline/ (see module docstring)."""
    decomposition = _load_decomposition(task_id, cache_dir)
    task = _load_task(task_id)
    agent_type = "external" if task["task"] in EXTERNAL_FACING_TASKS else "internal"
    subtasks = decomposition["subtasks"]
    subtasks_by_id = {s["id"]: s for s in subtasks}

    order = _topological_order(subtasks)
    schema_str = _build_schema(B2C_SCHEMA)
    sf_connector = SalesforceConnector(org_type=org_type)

    outputs_by_id = {}
    trace = []
    total_usage = _new_usage()

    try:
        for sid in order:
            subtask = subtasks_by_id[sid]
            context_block = _build_context_block(subtask, outputs_by_id, subtasks_by_id)
            role = subtask["role"]
            t0 = time.time()

            if role == "Retriever":
                output, turns, usage = _run_retriever_subtask(
                    subtask, task["query"], context_block, task.get("metadata"), agent_type,
                    schema_str, sf_connector, model, temperature, max_retriever_turns,
                )
            elif role == "Analyst":
                output, turns, usage = _run_analyst_subtask(subtask, task["query"], context_block, model, temperature)
            elif role == "Responder":
                output, turns, usage = _run_responder_subtask(subtask, task["query"], context_block, model, temperature)
            else:
                raise WorkflowError(f"Subtask {sid} has unknown role '{role}'")

            outputs_by_id[sid] = output
            _merge_usage(total_usage, usage)
            trace.append({
                "subtask_id": sid,
                "role": role,
                "description": subtask["description"],
                "depends_on": subtask["depends_on"],
                "context_received": context_block,
                "output": output,
                "turns": turns,
                "usage": usage,
                "wall_time_sec": round(time.time() - t0, 3),
            })

        final_sid = order[-1]
        final_output = outputs_by_id[final_sid]

        evaluator = Evaluator(model=EVALUATOR_MODEL, provider=EVALUATOR_PROVIDER)
        action_trajectory = [f"[{subtasks_by_id[sid]['role']}] {outputs_by_id[sid]}" for sid in order]
        reward_info = evaluator.evaluate(
            final_output, task["answer"], task["reward_metric"], task["task"], action_trajectory
        )
        reward = reward_info["reward"]
        total_cost = sum(c for c in total_usage["cost"] if c is not None)

        result = {
            "task_id": task_id,
            "run_number": run_number,
            "task_type": task["task"],
            "gt_answer": task["answer"],
            "reward": reward,
            "agent_info": {
                "architecture": "workflow",
                "execution_order": order,
                "final_subtask_id": final_sid,
                "end_reason": {
                    "source": "agent",
                    "message": "Workflow completed",
                    "content": final_output,
                    "parsed_answer": reward_info["parsed_answer"],
                },
                "usage": total_usage,
                "total_cost": total_cost,
                "num_subtasks": len(order),
            },
            "traj": trace,
        }
    except Exception as e:
        traceback.print_exc()
        result = {
            "task_id": task_id,
            "run_number": run_number,
            "task_type": task["task"],
            "gt_answer": task["answer"],
            "reward": 0,
            "agent_info": {
                "architecture": "workflow",
                "source": "api",
                "content": f"Error: {e}",
                "usage": total_usage,
                "total_cost": sum(c for c in total_usage["cost"] if c is not None),
            },
            "traj": trace,
        }

    return result


def save_result(result, log_dir=DEFAULT_LOG_DIR, model=DEFAULT_MODEL):
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"results_{model}_workflow_task{result['task_id']}_run{result['run_number']}.json")
    with open(out_path, "w") as f:
        json.dump([result], f, indent=2)
    return out_path


if __name__ == "__main__":
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 722
    run_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    result = run_workflow(task_id, run_number=run_number)
    path = save_result(result, model=DEFAULT_MODEL)
    print(f"Saved {path} -- reward={result['reward']}")
