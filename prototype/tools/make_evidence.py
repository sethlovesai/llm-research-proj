"""Builds data/bucket_evidence.json (task ids and categories per DAG width) from the repo's
decomposition cache, read-only, using the same width rule as router.load_tasks:
width = features.parallelism_width, or 1 when absent (undecomposed task)."""
import hashlib, json, os, glob, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "decomposition", "cache"))
widths = {}
digest = hashlib.sha256()
for p in sorted(glob.glob(os.path.join(CACHE, "*.json"))):
    raw = open(p, "rb").read(); digest.update(raw)
    d = json.loads(raw)
    w = int((d.get("features") or {}).get("parallelism_width") or 1)
    e = widths.setdefault(str(w), {"task_ids": [], "categories": []})
    e["task_ids"].append(d["task_id"]); e["categories"].append(d["category"])
for e in widths.values():
    e["task_ids"].sort(); e["n_tasks"] = len(e["task_ids"])
    e["categories"] = sorted(set(e["categories"])); e["n_categories"] = len(e["categories"])
out = {"source": "decomposition/cache/*.json (read-only)", "n_cache_files": len(glob.glob(os.path.join(CACHE, "*.json"))),
       "sha256_of_concatenated_cache_files_sorted_by_name": digest.hexdigest(),
       "built_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "widths": widths}
json.dump(out, open(os.path.join(HERE, "..", "data", "bucket_evidence.json"), "w"), indent=2)
print(json.dumps({k: (v["n_tasks"], v["n_categories"]) for k, v in widths.items()}), out["n_cache_files"])
