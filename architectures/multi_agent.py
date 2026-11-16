"""
Multi-agent supervisor architecture: executes a task's cached decomposition
(decomposition/cache/{task_id}.json) respecting the DAG's real parallel/
dependency structure -- independent subtasks run concurrently within the
same "wave" (dependency level), a supervisor waits for each wave to finish,
retries any subtask that fails once, then runs an explicit synthesis step
before handing off to the Responder.

Reuses workflow.py's per-subtask execution logic VERBATIM (same
_run_retriever_subtask / _run_analyst_subtask / _run_responder_subtask, same
prompts, same context-building via _build_context_block) so the only things
that differ from the workflow architecture are:
  (a) wave-based concurrent scheduling instead of a flat topological sort
      (see `_compute_levels` -- subtasks with a genuine sequential
      dependency, e.g. task 1857's Analyst -> Analyst chain, still land in
      separate levels and never run concurrently with each other; only
      subtasks that are mutually independent by construction share a level),
  (b) per-subtask retry-once-on-failure, with Analyst subtasks additionally
      classified into a tri-state outcome (satisfied / not_satisfied /
      unable_to_retrieve),
  (c) an explicit supervisor synthesis step, run once after all non-
      Responder subtasks finish, that judges whether the gathered
      information is sufficient before the Responder formats an answer.
(b) and (c) are deliberately NOT backported into workflow.py -- per-branch
failure isolation and cross-branch synthesis are capabilities specific to
having a supervisor over independent concurrent branches; a sequential
workflow doesn't have separate branches to isolate or synthesize across.

IMPORTANT CAVEAT: because (c) always runs -- even on a DAG with no real
parallelism to exploit (parallelism_width=1) -- it is an extra LLM call the
workflow architecture never makes. So width=1 sanity-check tasks are NOT
guaranteed a byte-identical final answer to workflow.py's; what should match
is the per-subtask outputs, execution structure (levels degenerate to single-
item levels, same effective order as workflow's topological sort), and
correctness. See test_multi_agent.py's validation output for how this is
checked and reported, rather than assumed.

Output schema mirrors workflow.py's run_workflow() (task_id, run_number,
task_type, gt_answer, reward, agent_info, traj) so results are directly
comparable and normalize_score.py works unchanged; traj gains "state" and
"attempts"/"failed_after_retry" fields per subtask, plus one extra trace
entry (role="Supervisor") for the synthesis step.
"""
import concurrent.futures
import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from litellm import completion

from architectures.workflow import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    EVALUATOR_MODEL,
    EVALUATOR_PROVIDER,
    DEFAULT_CACHE_DIR,
    MAX_RETRIEVER_TURNS,
    _load_decomposition,
    _load_task,
    _build_schema,
    _build_context_block,
    _new_usage,
    _merge_usage,
    _record_usage,
    _run_retriever_subtask,
    _run_analyst_subtask,
    _run_responder_subtask,
)
from crm_sandbox.data.assets import B2C_SCHEMA, EXTERNAL_FACING_TASKS
from crm_sandbox.env.connect_sandbox import SalesforceConnector
from crm_sandbox.env.env import Evaluator

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "multi_agent")

SUPERVISOR_SYNTHESIS_PROMPT = """\
You are the Supervisor coordinating a multi-agent CRM task pipeline. All \
independent subtasks have now finished (some possibly after one retry; any \
that still failed are marked accordingly below). Your job is to judge \
whether, taken together, there is enough information to answer the overall \
task, and to produce a concise synthesis for the final Responder step to use.

# Overall task
{task_query}

# Subtask results
{subtask_summary}

Respond in EXACTLY this format, nothing else:
<verdict>SUFFICIENT</verdict> or <verdict>INSUFFICIENT</verdict>
<synthesis>a concise synthesis of the relevant findings for the Responder to \
use, or -- if insufficient -- a clear statement of what's missing</synthesis>"""


class MultiAgentError(Exception):
    """Raised for multi-agent-execution failures not caused by an underlying LLM/tool error."""


def _sort_key(sid):
    try:
        return (0, int(sid))
    except ValueError:
        return (1, sid)


def _compute_levels(subtasks):
    """Groups subtasks into dependency waves: level 0 = subtasks with no
    dependencies, level k = subtasks whose dependencies all finished by
    level k-1. This is the multi-agent equivalent of workflow.py's flat
    topological sort -- except every subtask within a level is mutually
    independent by construction (if two subtasks in the same level depended
    on each other, one of them couldn't have reached indegree 0 yet), so
    they're safe to run concurrently with no ordering constraint between
    them."""
    subtasks_by_id = {s["id"]: s for s in subtasks}
    indegree = {sid: 0 for sid in subtasks_by_id}
    children = {sid: [] for sid in subtasks_by_id}
    for s in subtasks:
        for dep in s["depends_on"]:
            if dep not in subtasks_by_id:
                raise MultiAgentError(f"Subtask {s['id']} depends on unknown id '{dep}'")
            children[dep].append(s["id"])
            indegree[s["id"]] += 1

    levels = []
    remaining_indegree = dict(indegree)
    done = set()
    while len(done) < len(subtasks_by_id):
        ready = sorted(
            [sid for sid in subtasks_by_id if sid not in done and remaining_indegree[sid] == 0],
            key=_sort_key,
        )
        if not ready:
            raise MultiAgentError("Cycle detected -- cannot compute execution levels.")
        levels.append(ready)
        for sid in ready:
            done.add(sid)
            for child in children[sid]:
                remaining_indegree[child] -= 1
    return levels


def _classify_analyst_state(output):
    """Not every Analyst subtask is a yes/no check -- ANALYST_SYSTEM_PROMPT's
    'Yes, satisfied.../No, not satisfied...' framing fits verification-style
    subtasks (e.g. lead_qualification's BANT checks) but not computational
    ones (e.g. 'compute the average close time per agent'). A successful,
    non-yes/no result is still bucketed as "satisfied" (the subtask
    resolved) rather than inventing a fourth state."""
    stripped = output.strip().lower()
    if stripped.startswith("yes"):
        return "satisfied"
    if stripped.startswith("no"):
        return "not_satisfied"
    return "satisfied"


def _run_subtask_with_retry(role, run_fn, *args, **kwargs):
    """Runs a subtask; retries exactly once on failure -- an exception, or
    (Retriever-specific) its own max-turns-without-<respond> fallback output.
    Never raises: a subtask still failing after retry becomes a sentinel
    failure result rather than aborting the run, so one branch's failure
    doesn't take down the other concurrent branches in its wave -- the
    per-branch failure isolation a sequential workflow doesn't need."""
    attempts = 0
    last_error = None
    for attempts in (1, 2):
        try:
            output, turns, usage = run_fn(*args, **kwargs)
            if role == "Retriever" and isinstance(output, str) and output.startswith("Error:"):
                last_error = output
                if attempts == 1:
                    continue
                return output, turns, usage, attempts, True
            return output, turns, usage, attempts, False
        except Exception as e:
            last_error = str(e)
            if attempts == 1:
                continue
    failure_output = f"Error: subtask failed after retry: {last_error}"
    return failure_output, [], _new_usage(), attempts, True


def _execute_subtask(subtask, task, agent_type, context_block, schema_str, org_type,
                      model, temperature, max_retriever_turns):
    role = subtask["role"]
    t0 = time.time()

    if role == "Retriever":
        # Fresh connector per call -- SalesforceConnector's session isn't
        # verified thread-safe for concurrent use across threads in the
        # same wave.
        sf_connector = SalesforceConnector(org_type=org_type)
        output, turns, usage, attempts, failed = _run_subtask_with_retry(
            role, _run_retriever_subtask, subtask, task["query"], context_block,
            task.get("metadata"), agent_type, schema_str, sf_connector, model, temperature, max_retriever_turns,
        )
        state = None
    elif role == "Analyst":
        output, turns, usage, attempts, failed = _run_subtask_with_retry(
            role, _run_analyst_subtask, subtask, task["query"], context_block, model, temperature,
        )
        state = "unable_to_retrieve" if failed else _classify_analyst_state(output)
    elif role == "Responder":
        output, turns, usage, attempts, failed = _run_subtask_with_retry(
            role, _run_responder_subtask, subtask, task["query"], context_block, model, temperature,
        )
        state = None
    else:
        raise MultiAgentError(f"Subtask {subtask['id']} has unknown role '{role}'")

    return {
        "subtask_id": subtask["id"],
        "role": role,
        "description": subtask["description"],
        "depends_on": subtask["depends_on"],
        "context_received": context_block,
        "output": output,
        "state": state,
        "attempts": attempts,
        "failed_after_retry": failed,
        "turns": turns,
        "usage": usage,
        "wall_time_sec": round(time.time() - t0, 3),
    }


def _run_supervisor_synthesis(task_query, completed_traces, model, temperature):
    lines = []
    for t in completed_traces:
        state_note = f"  state={t['state']}" if t.get("state") else ""
        lines.append(
            f"--- Subtask {t['subtask_id']} ({t['role']}): {t['description']}{state_note} ---\n{t['output']}"
        )
    subtask_summary = "\n\n".join(lines)

    prompt = SUPERVISOR_SYNTHESIS_PROMPT.format(task_query=task_query, subtask_summary=subtask_summary)
    messages = [{"role": "user", "content": prompt}]
    usage = _new_usage()
    res = completion(messages=messages, model=model, temperature=temperature, max_tokens=1500, top_p=1.0)
    _record_usage(usage, res)
    raw = res.choices[0].message.content.strip()

    verdict_match = re.search(r"<verdict>\s*(SUFFICIENT|INSUFFICIENT)\s*</verdict>", raw, re.IGNORECASE)
    synthesis_match = re.search(r"<synthesis>(.*?)</synthesis>", raw, re.DOTALL)
    verdict = verdict_match.group(1).upper() if verdict_match else "SUFFICIENT"
    synthesis = synthesis_match.group(1).strip() if synthesis_match else raw

    return {"verdict": verdict, "synthesis": synthesis, "raw": raw}, usage


def run_multi_agent(task_id, run_number=1, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE,
                     cache_dir=DEFAULT_CACHE_DIR, max_retriever_turns=MAX_RETRIEVER_TURNS, org_type="b2c"):
    """Executes task_id's cached decomposition respecting the DAG's real
    parallel structure and scores the final Responder's output. Returns a
    result dict in the same schema as workflow.py's run_workflow() (see
    module docstring)."""
    decomposition = _load_decomposition(task_id, cache_dir)
    task = _load_task(task_id)
    agent_type = "external" if task["task"] in EXTERNAL_FACING_TASKS else "internal"
    subtasks = decomposition["subtasks"]
    subtasks_by_id = {s["id"]: s for s in subtasks}

    levels = _compute_levels(subtasks)
    schema_str = _build_schema(B2C_SCHEMA)

    outputs_by_id = {}
    trace = []
    total_usage = _new_usage()

    try:
        if not levels or len(levels[-1]) != 1:
            raise MultiAgentError(
                "Expected the final execution level to contain exactly one subtask (the DAG's "
                "unique sink -- guaranteed by decompose.py's validated DAGs)."
            )
        final_id = levels[-1][0]

        if len(levels) == 1:
            # Trivial single-subtask case (decomposition_warranted=false, e.g. task 722): no
            # upstream subtasks to run concurrently or synthesize over -- the lone subtask's role
            # need not be Responder (decompose.py doesn't require that when warranted=false).
            # Execute it directly, identical to how workflow.py handles this same case.
            subtask = subtasks_by_id[final_id]
            context_block = _build_context_block(subtask, outputs_by_id, subtasks_by_id)
            only_trace = _execute_subtask(
                subtask, task, agent_type, context_block, schema_str, org_type,
                model, temperature, max_retriever_turns,
            )
            outputs_by_id[final_id] = only_trace["output"]
            _merge_usage(total_usage, only_trace["usage"])
            trace.append(only_trace)
            synthesis_result = None
        else:
            for level in levels[:-1]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(level))) as executor:
                    futures = []
                    for sid in level:
                        subtask = subtasks_by_id[sid]
                        context_block = _build_context_block(subtask, outputs_by_id, subtasks_by_id)
                        futures.append(executor.submit(
                            _execute_subtask, subtask, task, agent_type, context_block, schema_str,
                            org_type, model, temperature, max_retriever_turns,
                        ))
                    level_traces = [f.result() for f in futures]

                for lt in level_traces:
                    outputs_by_id[lt["subtask_id"]] = lt["output"]
                    _merge_usage(total_usage, lt["usage"])
                    trace.append(lt)

            synthesis_result, synthesis_usage = _run_supervisor_synthesis(task["query"], trace, model, temperature)
            _merge_usage(total_usage, synthesis_usage)
            trace.append({
                "subtask_id": None,
                "role": "Supervisor",
                "description": "Synthesize all subtask outputs and judge whether there is enough information to answer.",
                "depends_on": [t["subtask_id"] for t in trace],
                "context_received": None,
                "output": synthesis_result["raw"],
                "state": synthesis_result["verdict"],
                "attempts": 1,
                "failed_after_retry": False,
                "turns": [],
                "usage": synthesis_usage,
                "wall_time_sec": None,
            })

            final_subtask = subtasks_by_id[final_id]
            base_context = _build_context_block(final_subtask, outputs_by_id, subtasks_by_id)
            synthesis_block = f"# Supervisor synthesis (verdict: {synthesis_result['verdict']})\n{synthesis_result['synthesis']}"
            combined_context = f"{base_context}\n\n{synthesis_block}" if base_context else synthesis_block

            final_trace = _execute_subtask(
                final_subtask, task, agent_type, combined_context, schema_str, org_type,
                model, temperature, max_retriever_turns,
            )
            outputs_by_id[final_id] = final_trace["output"]
            _merge_usage(total_usage, final_trace["usage"])
            trace.append(final_trace)

        final_output = outputs_by_id[final_id]

        evaluator = Evaluator(model=EVALUATOR_MODEL, provider=EVALUATOR_PROVIDER)
        action_trajectory = [f"[{t['role']}] {t['output']}" for t in trace]
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
                "architecture": "multi_agent",
                "execution_levels": levels,
                "final_subtask_id": final_id,
                "supervisor_synthesis": synthesis_result,
                "end_reason": {
                    "source": "agent",
                    "message": "Multi-agent run completed",
                    "content": final_output,
                    "parsed_answer": reward_info["parsed_answer"],
                },
                "usage": total_usage,
                "total_cost": total_cost,
                "num_subtasks": len(subtasks),
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
                "architecture": "multi_agent",
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
    out_path = os.path.join(log_dir, f"results_{model}_multi_agent_task{result['task_id']}_run{result['run_number']}.json")
    with open(out_path, "w") as f:
        json.dump([result], f, indent=2)
    return out_path


if __name__ == "__main__":
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 89
    run_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    result = run_multi_agent(task_id, run_number=run_number)
    path = save_result(result, model=DEFAULT_MODEL)
    print(f"Saved {path} -- reward={result['reward']}")
