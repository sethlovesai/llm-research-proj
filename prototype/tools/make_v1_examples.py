"""Builds data/v1_examples.json, read-only from the repo's decomposition cache.

Benchmark examples copy the stored query, decomposition_warranted flag and subtasks (id, role,
description, depends_on) verbatim from decomposition/cache/<task_id>.json, with the source file's
sha256. The stored `features` are kept ONLY for provenance and tests; the app recomputes width from
the subtasks. One illustrative example is authored here and labelled as such (it is not a benchmark
task and has no evaluated evidence)."""
import datetime, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "decomposition", "cache"))
OUT = os.path.join(HERE, "..", "data", "v1_examples.json")

BENCHMARK = [  # (key, short label)
    ("bench_900002", 900002, "Flag accounts with an open support case AND a recent order"),
    ("bench_89", 89, "Qualify a lead (BANT check)"),
    ("bench_2056", 2056, "Find the agent with the fastest sales cycle"),
    ("bench_722", 722, "Answer a knowledge-base question"),
]

examples = []
for key, tid, label in BENCHMARK:
    path = os.path.join(CACHE, f"{tid}.json")
    raw = open(path, "rb").read()
    d = json.loads(raw)
    assert d["task_id"] == tid
    examples.append({
        "key": key, "label": label, "origin": "benchmark_in_sample", "task_id": tid, "category": d["category"],
        "description": d["query"], "decomposition_warranted": d["decomposition_warranted"],
        "subtasks": [{"id": s["id"], "role": s["role"], "description": s["description"], "depends_on": list(s["depends_on"])}
                     for s in d["subtasks"]],
        "source": f"decomposition/cache/{tid}.json", "source_sha256": hashlib.sha256(raw).hexdigest(),
        "stored_features_for_provenance_only": d.get("features"),
    })

examples.append({
    "key": "illus_w3", "label": "Customer health check across three data sources", "origin": "illustrative",
    "task_id": None, "category": None,
    "description": ("For a given customer account, check its open support cases, its orders in the last six months "
                    "and its latest satisfaction survey score, then summarise whether the account needs attention."),
    "decomposition_warranted": True,
    "subtasks": [
        {"id": "1", "role": "Retriever", "description": "Retrieve the account's open support cases.", "depends_on": []},
        {"id": "2", "role": "Retriever", "description": "Retrieve the account's orders from the last six months.", "depends_on": []},
        {"id": "3", "role": "Retriever", "description": "Retrieve the account's latest satisfaction survey score.", "depends_on": []},
        {"id": "4", "role": "Analyst", "description": "Combine the three results and judge whether the account needs attention.",
         "depends_on": ["1", "2", "3"]},
        {"id": "5", "role": "Responder", "description": "Write a short summary with the judgement and the evidence.", "depends_on": ["4"]},
    ],
    "source": "authored for the prototype (hand-written DAG; not produced by the decomposition module, not evaluated)",
    "source_sha256": None, "stored_features_for_provenance_only": None,
})

json.dump({"built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "note": "Benchmark examples are IN-SAMPLE (part of the 24 tasks the model was fitted on).",
           "examples": examples}, open(OUT, "w"), indent=2)
print([(e["key"], len(e["subtasks"])) for e in examples])
