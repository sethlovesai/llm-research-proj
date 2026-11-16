"""
oracle.py -- retrospective best-architecture-per-task ("oracle") computation.

For each task_id (every task with a cached decomposition in decomposition/cache/,
currently 24: the 22 real CRMArena-Pro tasks + the 2 custom tasks 900001/900002),
looks at whichever of {single_agent, workflow, multi_agent} have result files on
disk and computes, per architecture, averaged across its valid (non-infra-error)
repeats:

    Q(A) = mean normalize_score.normalize_score() value (0-1)
    C(A) = mean agent_info.total_cost ($)
    L(A) = mean total wall-clock latency (sum of per-subtask wall_time_sec;
           None for single_agent -- see KNOWN DATA GAPS below)
    U(A) = Q(A) - lambda_ * C(A)        (lambda_ defaults to 0.0: pure-quality
                                          oracle, per current project decision)

The oracle's choice for a task is argmax_A U(A), restricted to architectures
that have at least one valid run for that task. Ties (common at this stage,
since many tasks score 0.0 under every architecture) are broken by lower cost,
then by a fixed priority order (single_agent < workflow < multi_agent) as a
final deterministic tiebreak -- documented here so it's never a silent
implementation detail.

A tiebreak pick is NOT a quality win. Tasks are classified on QUALITY (Q), never on
utility, so the classification is identical at every lambda: a task is "strict" if
exactly one architecture attains the best Q (within TIE_EPS), and "tied" if two or
more share it -- there is then no quality winner, and which architecture the
oracle "chose" is only a cost tiebreak. The win-count sections count only strict
tasks, attributing each to its best-Q architecture (the quality winner), NOT to the
oracle's pick: at lambda > 0 the oracle may pick a cheaper architecture than the
quality winner, which is a cost-quality tradeoff, not a quality win. Tied tasks
are reported in their own column (with an all-zero subcount). Raw oracle picks
(including cost-tiebreak and cost-driven ones) appear only as a labelled
reference line, never as the headline.

AGREED APPROACH FOR THE FORTHCOMING REGRET ANALYSIS (decided 2026-10-01;
specification only -- not yet implemented; the regret script should follow this
rather than re-deciding it):
  Regret is only meaningful where the oracle's choice reflects a genuine quality
  difference. Tasks where ALL available architectures are tied on QUALITY
  (mean normalized score within TIE_EPS of each other) -- in practice the
  all-zero-quality ties, where no architecture achieved meaningful quality and
  the oracle's pick is only a cost
  tiebreak used to give the oracle a defined reference pick -- are EXCLUDED from
  regret scoring. They are neither silently folded into the headline regret
  number nor silently dropped: the regret output reports them separately, e.g.
  "13 of 24 tasks excluded from regret scoring -- no architecture achieved
  meaningful quality; cost-tiebreak used for oracle's reference pick only",
  together with the excluded task_ids.
  Detection: oracle_for_task() sets `regret_excluded` -- True iff every available
  architecture has the same mean quality Q (max - min < TIE_EPS). This is tested
  on QUALITY, never on utility, so the excluded set does not change with lambda:
  for lambda > 0, cost differences make every utility distinct, which would
  otherwise turn the all-zero tasks into cost-driven "strict wins" and silently
  exclude nothing. The oracle's actual architecture PICK still maximizes utility
  at each lambda (cost-aware, as intended); only the exclusion test is
  quality-based. A task with a single available architecture is vacuously
  quality-tied and excluded (nothing to compare). A partial quality tie (two
  architectures share the best Q and a third is genuinely worse) is NOT
  excluded: a real quality gap exists and regret against it is well-defined. No
  such task exists in the current data (the 13 excluded are 3-way all-zero).
  Relationship to the strict/tied classification above: both are quality-based
  and therefore lambda-independent, but they are different tests -- "tied" means
  two or more architectures share the best Q (no unique quality winner), while
  "regret-excluded" means ALL available architectures share it. They coincide in
  the current data (no partial ties) but diverge for a partial tie, which is
  "tied" yet not regret-excluded.

Infra-error handling: a run whose agent_info.source == "api" (the same signal
CLAUDE.md's "Known gotchas" section documents -- a crashed/rate-limited run
silently recorded as reward=0) is excluded from an architecture's averages for
that task, not counted as a genuine zero. If every run for an architecture on a
task is an infra error, that architecture is treated as having NO data for that
task (same as if no result files existed), not as a valid zero-quality result.

KNOWN DATA GAPS (flagged explicitly here rather than silently worked around --
these are properties of the data that exists right now, not bugs in this
script):
  - Tiers come from task_tiers.py, an author-supplied mapping (not derived from
    any computed property of the tasks). A task_id missing from that mapping is
    reported as tier="unknown" rather than guessed.
  - Single-agent's saved result schema (results/custom_baseline/) has no
    wall-clock timing field at all -- only agent_info.num_turns, a turn COUNT,
    not a time unit. Latency is reported as None wherever single_agent is the
    oracle's pick; it is never approximated from turn count.
  - single_agent was never run on the two custom tasks (900001/900002), so those
    two tasks have only 2 architectures available. The per-task table's
    `available` column exists so this is never mistaken for a complete 3-way
    comparison.

Usage:
    python3 oracle.py                  # lambda_ = 0.0 (pure quality)
    python3 oracle.py --lambda 0.01    # quality-cost tradeoff
"""
import argparse
import glob
import json
import os

from statistics import variance

from normalize_score import normalize_score
from task_tiers import TIER_ORDER, get_tier

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(REPO_ROOT, "decomposition", "cache")

# Glob patterns (relative to REPO_ROOT), matching each architecture's actual
# save_result() filename convention verbatim.
RESULT_GLOBS = {
    "single_agent": "results/custom_baseline/results_gpt-4o-mini_react_task{tid}_run*.json",
    "workflow": "results/workflow/results_gpt-4o-mini_workflow_task{tid}_run*.json",
    "multi_agent": "results/multi_agent/results_gpt-4o-mini_multi_agent_task{tid}_run*.json",
}

ARCH_PRIORITY = {"single_agent": 0, "workflow": 1, "multi_agent": 2}

DEFAULT_LAMBDA = 0.0

# Utilities closer than this count as tied (guards against float noise in averages).
TIE_EPS = 1e-9


def all_task_ids():
    ids = []
    for path in glob.glob(os.path.join(CACHE_DIR, "*.json")):
        ids.append(int(os.path.basename(path)[:-5]))
    return sorted(ids)


def task_meta(task_id):
    with open(os.path.join(CACHE_DIR, f"{task_id}.json")) as f:
        d = json.load(f)
    features = d.get("features") or {}
    return {
        "category": d["category"],
        "tier": get_tier(task_id),
        "parallelism_width": features.get("parallelism_width"),
    }


def _load_runs(task_id, arch):
    pattern = RESULT_GLOBS[arch].format(tid=task_id)
    paths = sorted(glob.glob(os.path.join(REPO_ROOT, pattern)))
    runs = []
    for p in paths:
        try:
            with open(p) as f:
                result = json.load(f)[0]
        except (json.JSONDecodeError, IndexError, OSError) as e:
            # Tolerate a file mid-write by a concurrent batch run rather than
            # crashing the whole report -- just skip it and note the skip.
            print(f"  WARNING: could not read {p} ({e}) -- skipping this run")
            continue
        is_infra_error = result.get("agent_info", {}).get("source") == "api"
        runs.append({"path": p, "result": result, "infra_error": is_infra_error})
    return runs


def _latency_seconds(result, arch):
    if arch == "single_agent":
        return None  # not recorded in this schema -- see module docstring
    traj = result.get("traj", [])
    times = [t.get("wall_time_sec") for t in traj if t.get("wall_time_sec") is not None]
    return sum(times) if times else None


STRUCTURED_TASK_IDS = (900001, 900002)
METRICS = ("f1", "structured")


def _run_quality(result, task_id, metric):
    """Quality of one run. metric="f1" is the original normalize_score value for every
    task. metric="structured" replaces it with structured_score.py's deterministic score on
    the two custom tasks only (where F1 was found to reward ID/date overlap over count
    correctness); all other tasks keep normalize_score under either metric."""
    if metric == "structured" and task_id in STRUCTURED_TASK_IDS:
        import structured_score
        answer = str((result.get("agent_info", {}).get("end_reason") or {}).get("content", ""))
        return structured_score.structured_score(task_id, answer)["score"]
    return normalize_score(result)


def architecture_stats(task_id, arch, metric="f1"):
    """None if no result files exist at all for this architecture/task.
    Otherwise a dict with n_runs/n_infra_errors and averaged quality/cost/
    latency over the valid (non-infra-error) runs -- quality is None if every
    run that exists was an infra error (treated as "no data", not a zero)."""
    runs = _load_runs(task_id, arch)
    if not runs:
        return None

    valid = [r for r in runs if not r["infra_error"]]
    n_infra = len(runs) - len(valid)
    if not valid:
        return {"n_runs": 0, "n_infra_errors": n_infra, "quality": None, "cost": None, "latency": None,
                "run_qualities": []}

    qualities, costs, latencies = [], [], []
    for r in valid:
        result = r["result"]
        qualities.append(_run_quality(result, task_id, metric))
        cost = result.get("agent_info", {}).get("total_cost")
        if cost is not None:
            costs.append(cost)
        lat = _latency_seconds(result, arch)
        if lat is not None:
            latencies.append(lat)

    return {
        "n_runs": len(valid),
        "n_infra_errors": n_infra,
        "quality": sum(qualities) / len(qualities),
        "cost": (sum(costs) / len(costs)) if costs else None,
        "latency": (sum(latencies) / len(latencies)) if latencies else None,
        "run_qualities": qualities,
    }


def compute_utility(stats, lambda_):
    if stats is None or stats["quality"] is None:
        return None
    cost = stats["cost"] if stats["cost"] is not None else 0.0
    return stats["quality"] - lambda_ * cost


def oracle_for_task(task_id, lambda_=DEFAULT_LAMBDA, metric="f1"):
    meta = task_meta(task_id)
    per_arch = {}
    for arch in RESULT_GLOBS:
        stats = architecture_stats(task_id, arch, metric)
        per_arch[arch] = {"stats": stats, "utility": compute_utility(stats, lambda_)}

    available = [a for a, v in per_arch.items() if v["utility"] is not None]

    if not available:
        return {
            "task_id": task_id, **meta, "available_architectures": available,
            "oracle_architecture": None, "oracle_quality": None,
            "oracle_cost": None, "oracle_latency": None, "per_arch": per_arch,
            "tied_architectures": [], "is_strict_win": False, "quality_winner": None,
            "all_zero_tie": False, "regret_excluded": False,
            "margin": None, "margin_se": None, "margin_over_se": None,
        }

    def sort_key(a):
        u = per_arch[a]["utility"]
        c = per_arch[a]["stats"]["cost"]
        c = c if c is not None else float("inf")
        return (-u, c, ARCH_PRIORITY[a])

    best = sorted(available, key=sort_key)[0]
    best_stats = per_arch[best]["stats"]

    # Task classification (strict/tied) and regret exclusion are decided on QUALITY,
    # not utility, so both are independent of lambda (see module docstring). The
    # oracle's pick above is the only thing that maximizes utility.
    qualities = {a: per_arch[a]["stats"]["quality"] for a in available}
    best_quality = max(qualities.values())
    tied = sorted(
        (a for a, q in qualities.items() if best_quality - q < TIE_EPS),
        key=lambda a: ARCH_PRIORITY[a],
    )
    is_strict = len(tied) == 1
    quality_winner = tied[0] if is_strict else None
    all_zero_tie = (not is_strict) and best_quality < TIE_EPS
    regret_excluded = (best_quality - min(qualities.values())) < TIE_EPS

    # Margin of the best architecture over the runner-up (by mean quality), and that
    # margin in standard errors of the difference: SE = sqrt(var_w/n_w + var_s/n_s),
    # sample variances over each architecture's per-run qualities. None when there is
    # no runner-up or fewer than 2 runs on either side; inf when SE is 0 and the
    # margin is positive (e.g. 3/3 vs 0/3).
    # If several architectures tie for second place, the runner-up used is the one that
    # gives the smallest margin/SE (the most conservative comparison).
    margin = margin_se = margin_over_se = None
    ranked = sorted(available, key=lambda a: -qualities[a])
    if len(ranked) >= 2:
        w = ranked[0]
        second_q = qualities[ranked[1]]
        picks = []
        for s in (a for a in ranked[1:] if second_q - qualities[a] < TIE_EPS):
            m = qualities[w] - qualities[s]
            qw = per_arch[w]["stats"]["run_qualities"]
            qs = per_arch[s]["stats"]["run_qualities"]
            se = ratio = None
            if len(qw) >= 2 and len(qs) >= 2:
                se = (variance(qw) / len(qw) + variance(qs) / len(qs)) ** 0.5
                if se > 0:
                    ratio = m / se
                elif m > TIE_EPS:
                    ratio = float("inf")
            picks.append((m, se, ratio))
        with_ratio = [p for p in picks if p[2] is not None]
        margin, margin_se, margin_over_se = (min(with_ratio, key=lambda p: p[2]) if with_ratio else picks[0])

    return {
        "task_id": task_id,
        **meta,
        "available_architectures": available,
        "oracle_architecture": best,
        "oracle_quality": best_stats["quality"],
        "oracle_cost": best_stats["cost"],
        "oracle_latency": best_stats["latency"],
        "per_arch": per_arch,
        "tied_architectures": tied,
        "is_strict_win": is_strict,
        "quality_winner": quality_winner,
        "all_zero_tie": all_zero_tie,
        "regret_excluded": regret_excluded,
        "margin": margin,
        "margin_se": margin_se,
        "margin_over_se": margin_over_se,
    }


def _fmt(x, pct=False, money=False, sec=False):
    if x is None:
        return "N/A"
    if pct:
        return f"{x*100:.1f}%"
    if money:
        return f"${x:.4f}"
    if sec:
        return f"{x:.1f}s"
    return str(x)


def print_report(lambda_, metric="f1"):
    task_ids = all_task_ids()
    rows = [oracle_for_task(tid, lambda_=lambda_, metric=metric) for tid in task_ids]

    print("=" * 130)
    print(f"ORACLE REPORT  (lambda={lambda_}, U(A) = Q(A) - lambda * C(A), metric={metric}"
          + (" -- structured score on 900001/900002, normalize_score on all other tasks)" if metric == "structured"
             else " -- normalize_score everywhere)"))
    print("=" * 130)
    print()
    print(f"{'task_id':>8} {'category':32} {'tier':9} {'width':>5} {'oracle':12} "
          f"{'quality':>8} {'margin':>8} {'m/SE':>6} {'cost':>10} {'latency':>9} {'basis':9} {'available':9}")
    print("-" * 146)
    for r in rows:
        width = r["parallelism_width"] if r["parallelism_width"] is not None else "-"
        oracle_arch = r["oracle_architecture"] or "NONE"
        avail = ",".join(a.replace("single_agent", "SA").replace("workflow", "WF").replace("multi_agent", "MA")
                          for a in r["available_architectures"]) or "none"
        if r["oracle_architecture"] is None:
            basis = "n/a"
        elif not r["is_strict_win"]:
            basis = f"tie({len(r['tied_architectures'])})"
        elif r["oracle_architecture"] == r["quality_winner"]:
            basis = "strict"
        else:
            basis = "cost-pick"
        if not r["is_strict_win"] or r["margin"] is None:
            margin_s = "-"
            mse_s = "-"
        else:
            margin_s = f"{r['margin']*100:.1f}pp"
            mo = r["margin_over_se"]
            mse_s = "n/a" if mo is None else ("inf" if mo == float("inf") else f"{mo:.2f}")
        print(f"{r['task_id']:>8} {r['category']:32} {r['tier']:9} {str(width):>5} {oracle_arch:12} "
              f"{_fmt(r['oracle_quality'], pct=True):>8} {margin_s:>8} {mse_s:>6} "
              f"{_fmt(r['oracle_cost'], money=True):>10} "
              f"{_fmt(r['oracle_latency'], sec=True):>9} {basis:9} {avail:9}")
    print()
    print("margin = best minus second-best mean quality; m/SE = that margin in standard errors of the")
    print("difference (per-run variability, 3 repeats each). '-' for tied tasks, 'inf' when both sides have")
    print("zero variance (e.g. 3/3 vs 0/3), 'n/a' when fewer than 2 runs exist on either side.")
    print()
    print("basis (judged on QUALITY, independent of lambda):")
    print("  'strict'    = exactly one architecture has the best quality and the oracle picked it;")
    print("  'cost-pick' = one architecture has the best quality, but at this lambda the oracle picked")
    print("                a cheaper one (a cost-quality tradeoff, not a quality win);")
    print("  'tie(n)'    = n architectures share the best quality -- the 'oracle' column is then only a")
    print("                cost tiebreak pick and is NOT a quality win.")

    print()
    print("=" * 130)
    print("COVERAGE SUMMARY (how complete is this comparison right now)")
    print("=" * 130)
    n_by_count = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in rows:
        n_by_count[len(r["available_architectures"])] += 1
    print(f"Tasks with 3 architectures available: {n_by_count[3]}")
    print(f"Tasks with 2 architectures available: {n_by_count[2]}")
    print(f"Tasks with 1 architecture available:  {n_by_count[1]}")
    print(f"Tasks with 0 architectures available: {n_by_count[0]}")
    total_infra = 0
    for r in rows:
        for arch, v in r["per_arch"].items():
            if v["stats"]:
                total_infra += v["stats"].get("n_infra_errors", 0)
    print(f"Infra-error runs excluded from averages (across all tasks/architectures): {total_infra}")
    unknown_tier = [r["task_id"] for r in rows if r["tier"] == "unknown"]
    if unknown_tier:
        print()
        print(f"WARNING: {len(unknown_tier)} task(s) missing from task_tiers.py -> tier 'unknown': {unknown_tier}")

    print()
    print("=" * 130)
    print("ORACLE WIN COUNTS -- STRICT QUALITY WINS ONLY (ties are NOT counted as wins; quality-based, lambda-independent)")
    print("=" * 130)

    scored = [r for r in rows if r["oracle_architecture"] is not None]
    n_strict = sum(1 for r in scored if r["is_strict_win"])
    n_tied = len(scored) - n_strict
    n_all_zero = sum(1 for r in scored if r["all_zero_tie"])
    print(f"Strict quality wins: {n_strict} of {len(scored)} tasks.")
    print(f"Tied (no quality winner; oracle pick was a cost tiebreak only): {n_tied} of {len(scored)} "
          f"-- of which {n_all_zero} are all-architectures-0.0%.")
    excluded = [r for r in scored if r["regret_excluded"]]
    n_excl_zero = sum(1 for r in excluded if r["oracle_quality"] < TIE_EPS)
    print(f"Excluded from regret scoring: {len(excluded)} of {len(scored)} tasks -- all available architectures "
          f"tied on quality ({n_excl_zero} all-zero); oracle reference pick there is a cost tiebreak only.")
    print(f"  excluded task_ids: {[r['task_id'] for r in excluded]}")

    def win_counts(keyfn):
        counts = {}
        for r in scored:
            c = counts.setdefault(
                keyfn(r), {"single_agent": 0, "workflow": 0, "multi_agent": 0, "tied": 0, "all_zero": 0, "tasks": 0}
            )
            c["tasks"] += 1
            if r["is_strict_win"]:
                c[r["quality_winner"]] += 1
            else:
                c["tied"] += 1
                c["all_zero"] += int(r["all_zero_tie"])
        return counts

    def print_counts(counts, order_key, label):
        for key, c in sorted(counts.items(), key=order_key):
            print(f"  {label(key):20} tasks={c['tasks']:>2}   strict wins: single_agent={c['single_agent']}  "
                  f"workflow={c['workflow']}  multi_agent={c['multi_agent']}   |  tied={c['tied']} "
                  f"(all-zero={c['all_zero']})")

    print("\n-- Overall --")
    print_counts(win_counts(lambda r: "all"), lambda kv: 0, lambda k: "all tasks")

    print("\n-- By tier --")
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    print_counts(win_counts(lambda r: r["tier"]), lambda kv: tier_rank.get(kv[0], len(TIER_ORDER)),
                 lambda k: f"tier={k}")

    print("\n-- By parallelism_width --")
    print_counts(win_counts(lambda r: r["parallelism_width"]), lambda kv: (kv[0] is None, kv[0]),
                 lambda k: f"width={k if k is not None else 'trivial(None)'}")

    ref = {"single_agent": 0, "workflow": 0, "multi_agent": 0}
    for r in scored:
        ref[r["oracle_architecture"]] += 1
    print()
    print(f"Reference only -- raw oracle picks at lambda={lambda_}, INCLUDING cost-tiebreak and cost-driven picks "
          f"(do not read as quality wins):")
    print(f"  single_agent={ref['single_agent']}  workflow={ref['workflow']}  multi_agent={ref['multi_agent']}")
    n_cost_pick = sum(1 for r in scored if r["is_strict_win"] and r["oracle_architecture"] != r["quality_winner"])
    print(f"  of the {n_strict} strict-quality tasks, {n_cost_pick} have an oracle pick that differs from the "
          f"quality winner at this lambda (cost-driven).")

    return rows


def compare_metrics(lambda_=DEFAULT_LAMBDA):
    """Side-by-side oracle summary under metric="f1" and metric="structured", and which
    tasks change status between them. Only tasks 900001/900002 can differ (the structured
    metric is applied to those two; every other task is scored identically under both)."""
    tids = all_task_ids()
    by = {m: {r["task_id"]: r for r in (oracle_for_task(t, lambda_=lambda_, metric=m) for t in tids)} for m in METRICS}
    short = {"single_agent": "SA", "workflow": "WF", "multi_agent": "MA"}

    def counts(rows, keyfn=lambda r: "all"):
        out = {}
        for r in rows:
            if r["oracle_architecture"] is None:
                continue
            c = out.setdefault(keyfn(r), {"single_agent": 0, "workflow": 0, "multi_agent": 0, "tied": 0, "n": 0, "excl": 0})
            c["n"] += 1
            c["excl"] += int(r["regret_excluded"])
            if r["is_strict_win"]:
                c[r["quality_winner"]] += 1
            else:
                c["tied"] += 1
        return out

    def fmt(c):
        return (f"strict SA/WF/MA = {c['single_agent']}/{c['workflow']}/{c['multi_agent']}  tied={c['tied']}  "
                f"regret-excluded={c['excl']}  (n={c['n']})")

    print("=" * 130)
    print(f"ORACLE UNDER BOTH METRICS  (lambda={lambda_}; quality-based, so identical at every lambda)")
    print("  f1         = normalize_score everywhere (the original metric)")
    print("  structured = structured_score.py on 900001/900002; normalize_score on all other tasks")
    print("=" * 130)
    rows_by = {m: list(by[m].values()) for m in METRICS}
    print("\n-- Overall --")
    for m in METRICS:
        print(f"  {m:10} {fmt(counts(rows_by[m])['all'])}")
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    print("\n-- By tier --")
    for tier in TIER_ORDER:
        for m in METRICS:
            c = counts(rows_by[m], lambda r: r["tier"]).get(tier)
            if c:
                print(f"  tier={tier:7} {m:10} {fmt(c)}")
    print("\n-- By parallelism_width --")
    widths = sorted({r["parallelism_width"] for r in rows_by["f1"]}, key=lambda w: (w is None, w))
    for w in widths:
        for m in METRICS:
            c = counts(rows_by[m], lambda r: r["parallelism_width"]).get(w)
            if c:
                print(f"  width={str(w if w is not None else 'trivial'):8} {m:10} {fmt(c)}")

    def qline(r):
        return " | ".join(f"{short[a]} {r['per_arch'][a]['stats']['quality']*100:5.1f}%" for a in r["available_architectures"])

    def margin_s(r):
        if not r["is_strict_win"] or r["margin"] is None:
            return "-"
        mo = r["margin_over_se"]
        return f"{r['margin']*100:.1f}pp, m/SE " + ("n/a" if mo is None else "inf" if mo == float("inf") else f"{mo:.2f}")

    def status(r):
        if not r["is_strict_win"]:
            return f"tie({len(r['tied_architectures'])})"
        return f"strict winner {short[r['quality_winner']]}"

    print("\n-- Tasks whose result differs between the metrics (the two custom tasks are always shown) --")
    flips = []
    for t in tids:
        a, b = by["f1"][t], by["structured"][t]
        differs = (a["quality_winner"], a["is_strict_win"], a["regret_excluded"]) != \
                  (b["quality_winner"], b["is_strict_win"], b["regret_excluded"])
        if differs:
            flips.append(t)
        if differs or t in STRUCTURED_TASK_IDS:
            print(f"\n  task {t} {a['category']}  {'** FLIPS **' if differs else '(same status)'}")
            print(f"    f1         : {status(a):22} {qline(a)}   margin {margin_s(a)}   regret-excluded={a['regret_excluded']}")
            print(f"    structured : {status(b):22} {qline(b)}   margin {margin_s(b)}   regret-excluded={b['regret_excluded']}")
    print(f"\nTasks whose quality winner / strict-vs-tied / regret-exclusion status changed: {flips or 'none'}")
    return by


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lambda_", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--metric", choices=METRICS, default="f1",
                        help="f1 = normalize_score everywhere (default); structured = structured_score on 900001/900002")
    parser.add_argument("--compare-metrics", action="store_true",
                        help="side-by-side summary under both metrics and which tasks flip")
    args = parser.parse_args()
    if args.compare_metrics:
        compare_metrics(args.lambda_)
    else:
        print_report(args.lambda_, args.metric)
