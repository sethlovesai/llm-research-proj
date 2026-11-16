"""
router.py -- a feature-based architecture router, evaluated against the oracle.

WHAT IT DOES
Predicts the best architecture (single_agent / workflow / multi_agent) for a task from
features read off its CACHED DECOMPOSITION DAG only -- never the category label or tier --
and is evaluated leave-one-category-out (headline) and leave-one-task-out (secondary,
optimistic: same-category twins stay in training).

REUSE (nothing in oracle.py, the decomposition cache, existing results or structured_score.py
is changed): result loading, per-architecture quality/cost averaging, the quality-based
tie classification and the regret-exclusion flag all come from oracle.oracle_for_task().

FEATURES (cached DAG only)
  width              = decomposition features.parallelism_width (undecomposed -> 1)
  has_refusal_check  = the DAG has exactly one root subtask and it is an Analyst
  bucket             = (width, has_refusal_check)       no depth / subtask count / kNN
  NOTE has_refusal_check LEAKS THE CATEGORY LABEL: decompose_task() injects the refusal check
  in code for the three privacy categories (the LLM alone omitted it for 3 of 4 privacy tasks),
  so in the cache it equals "is a privacy category". Two settings are therefore evaluated:
    "true": the flag as cached (an upper bound -- a new task would not get the code injection
            without its category);
    "llm" : the flag as the LLM alone produced it = cached flag AND NOT llm_omitted_refusal_check
            (True only for 1624). Assumes the LLM-only DAG had the same width (the pre-injection
            DAG is not stored).

ESTIMATOR (EmpiricalShrinkageEstimator; swappable -- see the Estimator protocol below)
  Per task and architecture:
     advantage a_t(arch) = Q_t(arch) - mean over the three architectures of Q_t
     cost ratio r_t(arch) = C_t(arch) / mean over the three architectures of C_t
  Within a bucket: average within each category first, then across categories;
  n = number of DISTINCT categories (same-category twins are not independent evidence).
  Shrink toward "no difference":
     advantage_hat = n/(n+k) * bucket_mean_advantage                     (k = 2 by default)
     cost_ratio_hat = 1 + n/(n+k) * (bucket_mean_ratio - 1)
  Why advantages, not raw Q: the decision compares architectures on the SAME task, so a task's
  overall quality level cancels; shrinking toward per-architecture global means would instead
  import each architecture's overall rank into sparse buckets.
  Cost choice: dollar costs differ ~10x across tasks, and a new task's absolute cost is unknown,
  so only the RELATIVE cost of the architectures (ratio to the task's mean over architectures)
  is transferred, shrunk toward 1 (equal cost), and re-anchored at the global mean task cost
  C_bar = mean over training tasks of (mean over architectures of C):
     cost_hat(arch) = C_bar * cost_ratio_hat(arch)
  n = 0 (empty bucket) -> advantage 0 and ratio 1 for every architecture: NO EVIDENCE.
  NOTE ON k: the shrink factor w = n/(n+k) multiplies BOTH the advantage and the cost deviation
  of every architecture in a bucket by the same positive constant, so k never changes a
  bucket's argmax or its bootstrap confidence (verified numerically for k in 0.5..50 over the
  whole lambda grid). It only scales the reported magnitudes. It would matter only if the
  shrinkage target were non-zero (e.g. a pooled prior across buckets) or if buckets were
  compared with each other. Sensitivity row (c) is therefore identical by construction.

DECISION   U_hat(arch) = advantage_hat(arch) - lambda * cost_hat(arch);   pick argmax.
  (The task-level Q mean is common to all architectures, so it drops out of the argmax.)
  Exact ties (to 1e-9) are resolved by lowest cost_hat, then the fixed priority
  single_agent < workflow < multi_agent -- the same rule oracle.py uses. A tie, or an empty
  bucket, is a DEFAULT pick, not evidence; every results table counts them separately and also
  gives regret with ties resolved at random (averaged over the tied set), so the priority
  order never earns credit it did not earn.

CONFIDENCE  Monte Carlo: 5000 bootstrap resamples (fixed seed) of the bucket's CATEGORIES,
  recomputing the shrunk estimate and the decision each time; confidence = share of draws that
  pick the chosen architecture. It is NOT reported when the spread is unreliable: n <= 1
  (one category cannot give a spread), an empty bucket, or an exact tie. n = 2 is computed but
  flagged "weak".

EVALUATION  Train on all 24 tasks (the 13 all-zero ties supply cost evidence and pull bucket
  advantages toward 0). Score regret and oracle agreement only on the 11 strict-win tasks
  (regret_excluded = False in oracle.py); the 13 ties are listed separately. Regret at lambda
  = oracle utility - chosen utility, with actual per-task mean Q and C.
  Baselines: always single_agent / workflow / multi_agent, and the best fixed strategy at each
  lambda (chosen in-sample, so it is a hard-to-beat baseline). No global-mean-only baseline.

USAGE
  python3 router.py export [--metric structured|f1]   # results/summary/per_task_arch[_f1].csv
  python3 router.py run    [--k 2]                    # full evaluation -> results/router/*
Use the project venv (numpy).
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

import oracle

ARCHS = list(oracle.RESULT_GLOBS)  # single_agent, workflow, multi_agent
SHORT = {"single_agent": "SA", "workflow": "WF", "multi_agent": "MA"}
PRIORITY = oracle.ARCH_PRIORITY
REPO = oracle.REPO_ROOT
OUT_DIR = os.path.join(REPO, "results", "router")
SUMMARY_DIR = os.path.join(REPO, "results", "summary")

DEFAULT_K = 2
MC_DRAWS = 5000
SEED = 20261006
LAMBDA_MAIN = [0.0, 10.0, 100.0, 300.0]
LAMBDA_GRID = [0.0] + [float(x) for x in np.geomspace(1.0, 3000.0, 25)]
TIE_EPS = 1e-9
SETTINGS = ("true", "llm")
PROTOCOLS = ("LOCO", "LOTO")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


# --------------------------------------------------------------------------- data
@dataclass
class Task:
    tid: int
    category: str
    width: int
    refusal_true: bool
    refusal_llm: bool
    Q: dict
    C: dict
    n_runs: dict
    scored: bool
    quality_winner: str

    def bucket(self, setting):
        return (self.width, self.refusal_true if setting == "true" else self.refusal_llm)


def bucket_str(b):
    return f"(w{b[0]}, {'refusal' if b[1] else 'no-refusal'})"


def load_tasks(metric="structured"):
    """One Task per cached task. Quality/cost come from oracle.oracle_for_task (unchanged)."""
    tasks = []
    for tid in oracle.all_task_ids():
        r = oracle.oracle_for_task(tid, lambda_=0.0, metric=metric)
        Q, C, n = {}, {}, {}
        for a in ARCHS:
            s = r["per_arch"][a]["stats"]
            if not s or s["quality"] is None or s["cost"] is None:
                raise RuntimeError(f"task {tid}: missing data for {a}")
            Q[a], C[a], n[a] = s["quality"], s["cost"], s["n_runs"]
        with open(os.path.join(oracle.CACHE_DIR, f"{tid}.json")) as f:
            d = json.load(f)
        roots = [s for s in d["subtasks"] if not s["depends_on"]]
        width = (d.get("features") or {}).get("parallelism_width") or 1
        refusal_true = len(roots) == 1 and roots[0]["role"] == "Analyst"
        refusal_llm = refusal_true and not d.get("llm_omitted_refusal_check", False)
        tasks.append(Task(tid, d["category"], int(width), refusal_true, refusal_llm, Q, C, n,
                          scored=not r["regret_excluded"], quality_winner=r["quality_winner"]))
    return tasks


def export_quality_table(metric="structured"):
    """results/summary/per_task_arch.csv (structured, default) or per_task_arch_f1.csv."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    name = "per_task_arch.csv" if metric == "structured" else f"per_task_arch_{metric}.csv"
    path = os.path.join(SUMMARY_DIR, name)
    rows = []
    for tid in oracle.all_task_ids():
        r = oracle.oracle_for_task(tid, lambda_=0.0, metric=metric)
        src = "structured_score" if (metric == "structured" and tid in oracle.STRUCTURED_TASK_IDS) \
            else "normalize_score"
        for a in ARCHS:
            s = r["per_arch"][a]["stats"]
            rows.append({"task_id": tid, "category": r["category"], "architecture": a, "metric": src,
                         "quality": round(s["quality"], 6), "cost": round(s["cost"], 8), "n_runs": s["n_runs"]})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path, rows


# ----------------------------------------------------------------------- estimator
@dataclass
class Estimate:
    n: int
    categories: list
    adv_hat: np.ndarray
    cost_hat: np.ndarray
    adv_mean: np.ndarray
    adv_se: object          # None when n <= 1 (spread unreliable)
    rho_hat: np.ndarray
    n_informative: int = 0      # categories in the bucket with any non-zero advantage
    prior_only: bool = False    # empty bucket resolved by a prior (pooled-prior variant only)
    eff_informative: float = 0.0  # participation ratio of category influence (see _effective_n)


def _effective_n(advs):
    """Effective number of categories carrying the bucket's evidence: (sum m)^2 / sum m^2 with
    m_c = the largest |advantage| of any architecture in category c. 2 categories of very
    different size give ~1.8; k equal categories give k; zero evidence gives 0."""
    m = np.abs(advs).max(axis=1)
    s2 = float((m ** 2).sum())
    return float(m.sum() ** 2 / s2) if s2 > 0 else 0.0


class EmpiricalShrinkageEstimator:
    """Interface (swap in a formal Bayesian model by providing the same three methods):
         fit(train_tasks, setting) -> self
         estimate(bucket)          -> Estimate
         bootstrap(bucket, rng, draws) -> (adv_hat_draws, cost_hat_draws) or None when n < 2
    """
    name = "empirical-shrinkage"
    # picks that are defaults, not evidence (an empty bucket gives advantage 0 for everyone)
    default_flags = ("no evidence (empty bucket)", "tie in estimate")

    def __init__(self, k=DEFAULT_K):
        self.k = k

    def fit(self, train, setting):
        self.c_bar = mean(mean(t.C.values()) for t in train)
        by = defaultdict(lambda: defaultdict(list))
        for t in train:
            qm, cm = mean(t.Q.values()), mean(t.C.values())
            adv = np.array([t.Q[a] - qm for a in ARCHS])
            ratio = np.array([t.C[a] / cm for a in ARCHS])
            by[t.bucket(setting)][t.category].append((adv, ratio))
        self.buckets = {}
        for b, cats in by.items():
            names = sorted(cats)
            advs = np.array([np.mean([x[0] for x in cats[c]], axis=0) for c in names])
            ratios = np.array([np.mean([x[1] for x in cats[c]], axis=0) for c in names])
            self.buckets[b] = (names, advs, ratios)
        return self

    def _shrink(self, adv_mean, ratio_mean, n):
        w = n / (n + self.k)
        return w * adv_mean, 1.0 + w * (ratio_mean - 1.0)

    def estimate(self, bucket):
        if bucket not in self.buckets:
            return Estimate(0, [], np.zeros(3), self.c_bar * np.ones(3), np.zeros(3), None, np.ones(3))
        names, advs, ratios = self.buckets[bucket]
        n = len(names)
        adv_hat, rho = self._shrink(advs.mean(0), ratios.mean(0), n)
        se = advs.std(axis=0, ddof=1) / math.sqrt(n) if n >= 2 else None
        n_inf = int((np.abs(advs).max(axis=1) > TIE_EPS).sum())
        return Estimate(n, names, adv_hat, self.c_bar * rho, advs.mean(0), se, rho, n_inf, False, _effective_n(advs))

    def bootstrap(self, bucket, rng, draws=MC_DRAWS):
        if bucket not in self.buckets or len(self.buckets[bucket][0]) < 2:
            return None
        names, advs, ratios = self.buckets[bucket]
        n = len(names)
        idx = rng.integers(0, n, size=(draws, n))
        adv_hat, rho = self._shrink(advs[idx].mean(1), ratios[idx].mean(1), n)
        return adv_hat, self.c_bar * rho


# --------------------------------------------------------------------- predictions
@dataclass
class Pred:
    e: Estimate
    boot: object
    defaults: tuple = ("no evidence (empty bucket)", "tie in estimate")


def make_predictions(tasks, protocol, setting, k=DEFAULT_K, estimator=EmpiricalShrinkageEstimator):
    preds = {}
    for t in tasks:
        if protocol == "LOCO":
            train = [x for x in tasks if x.category != t.category]
        else:
            train = [x for x in tasks if x.tid != t.tid]
        est = estimator(k).fit(train, setting)
        b = t.bucket(setting)
        rng = np.random.default_rng([SEED, t.tid])
        preds[t.tid] = Pred(est.estimate(b), est.bootstrap(b, rng, MC_DRAWS), estimator.default_flags)
    return preds


def decide(adv_hat, cost_hat, lam):
    S = adv_hat - lam * cost_hat
    best = S.max()
    tied = [i for i in range(3) if best - S[i] < TIE_EPS]
    pick = sorted(tied, key=lambda i: (cost_hat[i], i))[0]
    return ARCHS[pick], [ARCHS[i] for i in tied], S


def evidence_flag(e, tied):
    if e.n == 0:
        return "prior only (empty bucket)" if e.prior_only else "no evidence (empty bucket)"
    if len(tied) > 1:
        return "tie in estimate"
    if e.n == 1:
        return "n=1 (spread unreliable)"
    if e.n == 2:
        return "n=2 (weak)"
    return "ok"


DEFAULT_FLAGS = ("no evidence (empty bucket)", "tie in estimate")


def actual_U(t, a, lam):
    return t.Q[a] - lam * t.C[a]


def oracle_pick(t, lam):
    """Same rule as oracle.py: max utility, ties by lower cost then fixed priority."""
    return sorted(ARCHS, key=lambda a: (-actual_U(t, a, lam), t.C[a], PRIORITY[a]))[0]


def task_result(t, pred, lam):
    pick, tied, _ = decide(pred.e.adv_hat, pred.e.cost_hat, lam)
    op = oracle_pick(t, lam)
    ou = actual_U(t, op, lam)
    flag = evidence_flag(pred.e, tied)
    conf = tie_share = conf_tb = None
    if flag in ("ok", "n=2 (weak)") and pred.boot is not None:
        adv_d, cost_d = pred.boot
        S_d = adv_d - lam * cost_d
        arg = S_d.argmax(axis=1)   # exact ties go to the first index (single_agent)
        conf = float((arg == ARCHS.index(pick)).mean())
        # diagnostics (not used by the original outputs): share of draws whose top score is an
        # exact tie, and the confidence if those ties are resolved by the point-decision rule
        # (lowest cost, then priority) instead of by index order
        top = np.sort(S_d, axis=1)
        tie_share = float(((top[:, -1] - top[:, -2]) < TIE_EPS).mean())
        tied_mask = (S_d.max(axis=1, keepdims=True) - S_d) < TIE_EPS
        conf_tb = float((np.where(tied_mask, cost_d, np.inf).argmin(axis=1) == ARCHS.index(pick)).mean())
    return {
        "tid": t.tid, "category": t.category, "scored": t.scored, "pick": pick, "tied": tied,
        "oracle": op, "flag": flag, "confidence": conf, "n": pred.e.n,
        "is_default": flag in pred.defaults, "tie_draw_share": tie_share, "confidence_tiebreak": conf_tb,
        "n_informative": pred.e.n_informative, "eff_informative": pred.e.eff_informative,
        "regret": ou - actual_U(t, pick, lam),
        "regret_tie": mean(ou - actual_U(t, a, lam) for a in tied),
        "agree": pick == op, "agree_tie": float(op in tied) / len(tied),
        "oracle_U": ou, "U": {a: actual_U(t, a, lam) for a in ARCHS},
        "adv_hat": pred.e.adv_hat.copy(),
    }


def fixed_regrets(tasks, lam, scored_ids):
    out = {a: {} for a in ARCHS}
    for t in tasks:
        if t.tid in scored_ids:
            ou = actual_U(t, oracle_pick(t, lam), lam)
            for a in ARCHS:
                out[a][t.tid] = ou - actual_U(t, a, lam)
    return out


def aggregate(results, tasks, lam, scored_ids):
    sc = [r for r in results if r["tid"] in scored_ids]
    fx = fixed_regrets(tasks, lam, scored_ids)
    fixed = {a: mean(fx[a].values()) for a in ARCHS}
    best = min(ARCHS, key=lambda a: (fixed[a], PRIORITY[a]))
    router, router_tie = mean(r["regret"] for r in sc), mean(r["regret_tie"] for r in sc)
    return {
        "n_scored": len(sc), "router": router, "router_tie": router_tie,
        "agree": sum(r["agree"] for r in sc), "agree_tie": sum(r["agree_tie"] for r in sc),
        "n_default": sum(r["is_default"] for r in sc),
        "n_prior_only": sum(r["flag"].startswith("prior only") for r in sc),
        "fixed": fixed, "best_fixed": best, "best_fixed_regret": fixed[best],
        "diff": router - fixed[best], "diff_tie": router_tie - fixed[best],
    }


def evaluate(tasks, preds, lam, scored_ids=None):
    if scored_ids is None:
        scored_ids = {t.tid for t in tasks if t.scored}
    results = [task_result(t, preds[t.tid], lam) for t in tasks]
    return results, aggregate(results, tasks, lam, scored_ids)


# ---------------------------------------------------------------------- break-even
def breakeven(t):
    """First lambda at which the oracle's pick moves off the quality winner: for each cheaper
    architecture c, lambda_c = (Q_w - Q_c) / (C_w - C_c); lambda* = min over c. None if the
    quality winner is also the cheapest."""
    w = t.quality_winner
    cands = []
    for c in ARCHS:
        if c != w and t.C[c] < t.C[w]:
            cands.append(((t.Q[w] - t.Q[c]) / (t.C[w] - t.C[c]), c, t.Q[w] - t.Q[c], t.C[w] - t.C[c]))
    return min(cands) if cands else None


# -------------------------------------------------------------------- bootstrap CI
def bootstrap_ci(tasks, results, lam, scored_ids, use_tie=False, draws=MC_DRAWS):
    """95% CI of (router regret - best fixed regret), resampling CATEGORIES of the scored
    tasks. The best fixed strategy is re-selected inside each resample."""
    key = "regret_tie" if use_tie else "regret"
    cats = sorted({t.category for t in tasks if t.tid in scored_ids})
    fx = fixed_regrets(tasks, lam, scored_ids)
    rr = {r["tid"]: r[key] for r in results}
    cnt = np.array([sum(1 for t in tasks if t.tid in scored_ids and t.category == c) for c in cats], float)
    rsum = np.array([sum(rr[t.tid] for t in tasks if t.tid in scored_ids and t.category == c) for c in cats])
    fsum = np.array([[sum(fx[a][t.tid] for t in tasks if t.tid in scored_ids and t.category == c) for c in cats]
                     for a in ARCHS])
    rng = np.random.default_rng([SEED, 7])
    idx = rng.integers(0, len(cats), size=(draws, len(cats)))
    W = np.stack([(idx == j).sum(1) for j in range(len(cats))], axis=1).astype(float)  # (draws, ncat)
    # elementwise sums rather than matmul: numpy 2.0.1 + Apple Accelerate emits spurious
    # floating-point warnings from matmul on this machine
    n_tot = (W * cnt).sum(axis=1)
    router = (W * rsum).sum(axis=1) / n_tot
    fixed = np.stack([(W * fsum[i]).sum(axis=1) / n_tot for i in range(3)], axis=1)
    diff = router - fixed.min(axis=1)
    return {"lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5)),
            "p_router_better": float((diff < 0).mean()), "n_categories": len(cats)}


# ----------------------------------------------------------------------- rendering
def render(headers, rows):
    cols = list(zip(*([headers] + [[str(c) for c in r] for r in rows]))) if rows else [[h] for h in headers]
    widths = [max(len(str(x)) for x in col) for col in cols]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    return [line(headers), "  ".join("-" * w for w in widths)] + [line([str(c) for c in r]) for r in rows]


def pct(x):
    return f"{x * 100:.1f}%"


def f3(x):
    return f"{x:.3f}"


# --------------------------------------------------------------------------- run
def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(k=DEFAULT_K):
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    tasks = load_tasks("structured")
    by_tid = {t.tid: t for t in tasks}
    scored_ids = {t.tid for t in tasks if t.scored}
    scored = [t for t in tasks if t.scored]
    ties = [t for t in tasks if not t.scored]

    path, _ = export_quality_table("structured")
    export_quality_table("f1")

    # ---- consistency with oracle.py
    for lam in LAMBDA_MAIN:
        for t in tasks:
            ref = oracle.oracle_for_task(t.tid, lambda_=lam, metric="structured")["oracle_architecture"]
            assert oracle_pick(t, lam) == ref, (t.tid, lam)

    emit("=" * 118)
    emit("ROUTER REPORT   estimator=empirical-shrinkage  k=%s  metric=structured (900001/900002; others normalize_score)" % k)
    emit("=" * 118)
    emit(f"quality table: {os.path.relpath(path, REPO)}   (oracle reuse check: router's oracle pick == oracle.py at lambda "
         f"{LAMBDA_MAIN} for all {len(tasks)} tasks: OK)")
    wins = Counter(t.quality_winner for t in scored)
    emit(f"scored (strict-win) tasks: {len(scored)}  single/workflow/multi = "
         f"{wins['single_agent']}/{wins['workflow']}/{wins['multi_agent']};  regret-excluded ties: {len(ties)}")

    # ---- 1. data
    emit("\n1. SCORED TASKS (11) AND FEATURES")
    rows = []
    for t in sorted(scored, key=lambda x: x.tid):
        rows.append([t.tid, t.category, t.width, "T" if t.refusal_true else "F", "T" if t.refusal_llm else "F",
                     bucket_str(t.bucket("true")), t.quality_winner] +
                    [f"{pct(t.Q[a])}/${t.C[a]:.4f}" for a in ARCHS])
    for l in render(["task", "category", "w", "refusal(true)", "refusal(llm)", "bucket(true)", "oracle@0",
                     "SA Q/C", "WF Q/C", "MA Q/C"], rows):
        emit("  " + l)
    emit("  Regret-excluded ties (13, listed separately; scored nowhere): " +
         ", ".join(f"{t.tid}" for t in sorted(ties, key=lambda x: x.tid)))
    emit("  Note: 900001 is effectively decisions-only -- every stored answer was a one-line decision, the three evidence")
    emit("  claims scored 0 for all nine, so its observed ceiling is 0.4.")

    # ---- 2. model card
    emit("\n2. WHAT THE ESTIMATOR LEARNS (fit on all 24 tasks, k=%s)" % k)
    for setting in SETTINGS:
        est = EmpiricalShrinkageEstimator(k).fit(tasks, setting)
        emit(f"  refusal flag = {setting}   (C_bar = ${est.c_bar:.5f})")
        rows = []
        for b in sorted(est.buckets):
            e = est.estimate(b)
            se = "unreliable (n<=1)" if e.adv_se is None else "/".join(f"{x:.3f}" for x in e.adv_se)
            rows.append([bucket_str(b), e.n, ",".join(e.categories) if e.n <= 3 else f"{e.n} categories",
                         "/".join(f"{x:+.3f}" for x in e.adv_hat), "/".join(f"{x:.2f}" for x in e.rho_hat), se])
        for l in render(["bucket", "n cats", "categories", "advantage_hat SA/WF/MA", "cost ratio_hat SA/WF/MA",
                         "SE of advantage (across categories)"], rows):
            emit("    " + l)

    # ---- predictions
    P = {(p, s): make_predictions(tasks, p, s, k) for p in PROTOCOLS for s in SETTINGS}

    # ---- 3. headline tables
    emit("\n3. RESULTS ON THE 11 SCORED TASKS  (regret = oracle utility - chosen utility; lower is better)")
    emit("   LOCO = leave-one-category-out (HEADLINE).  LOTO = leave-one-task-out (OPTIMISTIC: same-category twins stay in training).")
    emit("   'regret*' resolves default picks (empty bucket / tie) at random instead of by the fixed priority order.")
    emit("   'default' = scored tasks whose pick is a default (no evidence or tie), not evidence.")
    summary_rows = []
    for protocol in PROTOCOLS:
        for setting in SETTINGS:
            label = f"{protocol}{' (headline)' if protocol == 'LOCO' else ' (optimistic)'}, refusal={setting}" + \
                    (" [upper bound]" if setting == "true" else " [LLM-only flag]")
            emit(f"\n  {label}")
            rows = []
            for lam in LAMBDA_MAIN:
                res, ag = evaluate(tasks, P[(protocol, setting)], lam)
                rows.append([f"{lam:g}", f3(ag["router"]), f3(ag["router_tie"]), f"{ag['agree']}/{ag['n_scored']}",
                             f"{ag['agree_tie']:.1f}", ag["n_default"], f3(ag["fixed"]["single_agent"]),
                             f3(ag["fixed"]["workflow"]), f3(ag["fixed"]["multi_agent"]),
                             f"{SHORT[ag['best_fixed']]} {f3(ag['best_fixed_regret'])}",
                             f"{ag['diff']:+.3f}", f"{ag['diff_tie']:+.3f}"])
                summary_rows.append({"protocol": protocol, "refusal_setting": setting, "lambda": lam, "k": k,
                                     "router_regret": round(ag["router"], 6), "router_regret_ties_averaged": round(ag["router_tie"], 6),
                                     "oracle_agreement": ag["agree"], "oracle_agreement_ties_averaged": round(ag["agree_tie"], 3),
                                     "n_default_picks": ag["n_default"], "n_scored": ag["n_scored"],
                                     "always_single_agent": round(ag["fixed"]["single_agent"], 6),
                                     "always_workflow": round(ag["fixed"]["workflow"], 6),
                                     "always_multi_agent": round(ag["fixed"]["multi_agent"], 6),
                                     "best_fixed": ag["best_fixed"], "best_fixed_regret": round(ag["best_fixed_regret"], 6),
                                     "router_minus_best_fixed": round(ag["diff"], 6),
                                     "router_minus_best_fixed_ties_averaged": round(ag["diff_tie"], 6)})
            for l in render(["lambda", "router", "router*", "agree", "agree*", "default", "always SA", "always WF",
                             "always MA", "best fixed", "router-best", "router*-best"], rows):
                emit("    " + l)
    write_csv(os.path.join(OUT_DIR, "summary_main.csv"), summary_rows)

    # ---- 4. width-2 behaviour
    emit("\n4. WIDTH-2 TASKS (900001 churn_risk, 900002 dual_signal) UNDER LEAVE-ONE-CATEGORY-OUT")
    emit("   Holding one out leaves the other as the ONLY width-2 category (n=1), so each is predicted from its sibling alone.")
    for setting in SETTINGS:
        emit(f"\n  refusal={setting}")
        rows = []
        for lam in LAMBDA_MAIN:
            res, ag = evaluate(tasks, P[("LOCO", setting)], lam)
            rmap = {r["tid"]: r for r in res}
            tot = sum(r["regret"] for r in res if r["tid"] in scored_ids)
            for tid in (900001, 900002):
                r = rmap[tid]
                share = r["regret"] / tot if tot > 0 else 0.0
                rows.append([f"{lam:g}", tid, f"{r['pick']} ({r['flag']})", r["oracle"], f3(r["regret"]), pct(share)])
        for l in render(["lambda", "task", "router pick (evidence)", "oracle pick", "regret", "share of router's total regret"], rows):
            emit("    " + l)

    # ---- 5. routing tables
    routing_rows = []
    for protocol in PROTOCOLS:
        for setting in SETTINGS:
            for lam in LAMBDA_MAIN:
                res, _ = evaluate(tasks, P[(protocol, setting)], lam)
                for r in res:
                    t = by_tid[r["tid"]]
                    routing_rows.append({
                        "protocol": protocol, "refusal_setting": setting, "lambda": lam, "k": k,
                        "task_id": t.tid, "category": t.category, "scored": t.scored, "width": t.width,
                        "has_refusal_check": t.refusal_true if setting == "true" else t.refusal_llm,
                        "bucket": bucket_str(t.bucket(setting)), "n_categories": r["n"],
                        "chosen_architecture": r["pick"], "tied_set": "|".join(r["tied"]),
                        "evidence": r["flag"], "confidence": "" if r["confidence"] is None else round(r["confidence"], 4),
                        "oracle_architecture": r["oracle"], "regret": round(r["regret"], 6),
                        "regret_ties_averaged": round(r["regret_tie"], 6),
                        "advantage_hat_SA": round(float(r["adv_hat"][0]), 6), "advantage_hat_WF": round(float(r["adv_hat"][1]), 6),
                        "advantage_hat_MA": round(float(r["adv_hat"][2]), 6),
                        "Q_SA": round(t.Q["single_agent"], 6), "Q_WF": round(t.Q["workflow"], 6), "Q_MA": round(t.Q["multi_agent"], 6)})
    write_csv(os.path.join(OUT_DIR, "routing_table.csv"), routing_rows)
    emit("\n5. PER-TASK ROUTING TABLE  (leave-one-category-out; refusal=true flag; lambda=0 and 100; all 24 tasks, scored ones starred)")
    for lam in (0.0, 100.0):
        res, _ = evaluate(tasks, P[("LOCO", "true")], lam)
        emit(f"\n  lambda={lam:g}")
        rows = []
        for r in sorted(res, key=lambda x: (not x["scored"], x["tid"])):
            t = by_tid[r["tid"]]
            conf = "n/a" if r["confidence"] is None else f"{r['confidence']*100:.0f}%"
            rows.append([("*" if t.scored else " ") + str(t.tid), t.category[:26], t.width, "T" if t.refusal_true else "F",
                         bucket_str(t.bucket("true")), r["n"], SHORT[r["pick"]], conf, r["flag"], SHORT[r["oracle"]],
                         f3(r["regret"]) if t.scored else "(tie)"])
        for l in render(["task", "category", "w", "ref", "bucket", "n cats", "router", "confidence", "evidence", "oracle", "regret"], rows):
            emit("    " + l)

    # ---- 6. lambda
    emit("\n6. LAMBDA SWEEP (log-spaced grid 0, 1 ... 3000; per-task picks/utilities saved to results/router/lambda_sweep.csv)")
    sweep_rows, curve_rows = [], []
    for protocol, setting in (("LOCO", "true"), ("LOCO", "llm"), ("LOTO", "true")):
        for lam in LAMBDA_GRID:
            res, ag = evaluate(tasks, P[(protocol, setting)], lam)
            curve_rows.append({"protocol": protocol, "refusal_setting": setting, "lambda": round(lam, 4),
                               "router": round(ag["router"], 6), "router_ties_averaged": round(ag["router_tie"], 6),
                               "always_single_agent": round(ag["fixed"]["single_agent"], 6),
                               "always_workflow": round(ag["fixed"]["workflow"], 6),
                               "always_multi_agent": round(ag["fixed"]["multi_agent"], 6),
                               "best_fixed": ag["best_fixed"], "best_fixed_regret": round(ag["best_fixed_regret"], 6),
                               "router_minus_best_fixed": round(ag["diff"], 6)})
            if protocol == "LOCO":
                for r in res:
                    sweep_rows.append({"refusal_setting": setting, "lambda": round(lam, 4), "task_id": r["tid"],
                                       "scored": r["scored"], "router_pick": r["pick"], "evidence": r["flag"],
                                       "oracle_pick": r["oracle"], "regret": round(r["regret"], 6),
                                       "U_single_agent": round(r["U"]["single_agent"], 6), "U_workflow": round(r["U"]["workflow"], 6),
                                       "U_multi_agent": round(r["U"]["multi_agent"], 6)})
    write_csv(os.path.join(OUT_DIR, "lambda_sweep.csv"), sweep_rows)
    write_csv(os.path.join(OUT_DIR, "lambda_curve.csv"), curve_rows)
    emit("  Router (LOCO, true flag) vs best fixed strategy across the grid:")
    rows = []
    for c in curve_rows:
        if c["protocol"] == "LOCO" and c["refusal_setting"] == "true":
            rows.append([f"{c['lambda']:g}", f3(c["router"]), SHORT[c["best_fixed"]], f3(c["best_fixed_regret"]),
                         f"{c['router_minus_best_fixed']:+.3f}"])
    for l in render(["lambda", "router regret", "best fixed", "its regret", "router - best fixed"], rows):
        emit("    " + l)

    be_rows = []
    emit("\n  Per-task break-even lambda* (scored tasks): first lambda at which the oracle's pick moves off the quality winner")
    rows = []
    for t in sorted(scored, key=lambda x: x.tid):
        b = breakeven(t)
        if b:
            lam_star, c, dq, dc = b
            rows.append([t.tid, t.category, SHORT[t.quality_winner], f"{lam_star:,.0f}", SHORT[c], f"{dq:.3f}", f"{dc*1e3:.3f}"])
            be_rows.append({"task_id": t.tid, "category": t.category, "quality_winner": t.quality_winner,
                            "breakeven_lambda": round(lam_star, 3), "flips_to": c, "delta_Q": round(dq, 6), "delta_C": round(dc, 8)})
        else:
            rows.append([t.tid, t.category, SHORT[t.quality_winner], "none (winner is cheapest)", "-", "-", "-"])
            be_rows.append({"task_id": t.tid, "category": t.category, "quality_winner": t.quality_winner,
                            "breakeven_lambda": "", "flips_to": "", "delta_Q": "", "delta_C": ""})
    for l in render(["task", "category", "quality winner", "lambda*", "flips to", "dQ", "dC ($e-3)"], rows):
        emit("    " + l)
    write_csv(os.path.join(OUT_DIR, "breakeven_lambda.csv"), be_rows)

    # ---- 7. uncertainty
    emit("\n7. UNCERTAINTY: bootstrap over categories, 95% CI of (router regret - best fixed regret)   [negative = router better]")
    emit("   %d resamples of the %d categories that contain a scored task; the best fixed strategy is re-selected in every resample." %
         (MC_DRAWS, len({t.category for t in scored})))
    ci_rows, rows = [], []
    for setting in SETTINGS:
        for lam in LAMBDA_MAIN:
            res, ag = evaluate(tasks, P[("LOCO", setting)], lam)
            ci = bootstrap_ci(tasks, res, lam, scored_ids)
            ci_t = bootstrap_ci(tasks, res, lam, scored_ids, use_tie=True)
            rows.append([setting, f"{lam:g}", f"{ag['diff']:+.3f}", f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]", f"{ci['p_router_better']*100:.0f}%",
                         f"[{ci_t['lo']:+.3f}, {ci_t['hi']:+.3f}]"])
            ci_rows.append({"refusal_setting": setting, "lambda": lam, "router_minus_best_fixed": round(ag["diff"], 6),
                            "ci_lo": round(ci["lo"], 6), "ci_hi": round(ci["hi"], 6), "p_router_better": round(ci["p_router_better"], 4),
                            "ci_lo_ties_averaged": round(ci_t["lo"], 6), "ci_hi_ties_averaged": round(ci_t["hi"], 6)})
    for l in render(["refusal", "lambda", "router-best", "95% CI", "P(router better)", "95% CI (ties averaged)"], rows):
        emit("    " + l)
    write_csv(os.path.join(OUT_DIR, "bootstrap_ci.csv"), ci_rows)

    # ---- sensitivity
    emit("\n   SENSITIVITY (leave-one-category-out, refusal=true; router-minus-best-fixed regret; scoring-set changes do not alter training)")
    sens_rows, rows = [], []

    def sens(label, tasks_v, preds_v, scored_v):
        out = [label]
        for lam in LAMBDA_MAIN:
            _, ag = evaluate(tasks_v, preds_v, lam, scored_v)
            out.append(f"{ag['diff']:+.3f} ({ag['router']:.3f} vs {SHORT[ag['best_fixed']]} {ag['best_fixed_regret']:.3f})")
            sens_rows.append({"variant": label, "lambda": lam, "router": round(ag["router"], 6), "best_fixed": ag["best_fixed"],
                              "best_fixed_regret": round(ag["best_fixed_regret"], 6), "router_minus_best_fixed": round(ag["diff"], 6),
                              "n_scored": ag["n_scored"]})
        rows.append(out)

    base_p = P[("LOCO", "true")]
    sens("baseline (k=%s)" % k, tasks, base_p, scored_ids)
    sens("(a) drop 722 from scoring", tasks, base_p, scored_ids - {722})
    sens("(b) drop 900001 and 900002 from scoring", tasks, base_p, scored_ids - {900001, 900002})
    for kk in (1, 2, 5):
        sens(f"(c) k={kk}", tasks, make_predictions(tasks, "LOCO", "true", kk), scored_ids)
    tasks_f1 = load_tasks("f1")
    sens("(d) F1 metric instead of structured", tasks_f1, make_predictions(tasks_f1, "LOCO", "true", k),
         {t.tid for t in tasks_f1 if t.scored})
    for l in render(["variant"] + [f"lambda={l:g}" for l in LAMBDA_MAIN], rows):
        emit("    " + l)
    emit("   Note: the three (c) rows are identical BY CONSTRUCTION, not by accident. The shrink factor n/(n+k) multiplies the")
    emit("   advantage and the cost deviation of every architecture in a bucket by the same constant, so k cannot change which")
    emit("   architecture wins a bucket (or the bootstrap confidence); it only scales the reported magnitudes. k would matter")
    emit("   only with a non-zero shrinkage target (e.g. a pooled prior across buckets).")
    write_csv(os.path.join(OUT_DIR, "sensitivity.csv"), sens_rows)

    emit("\n   (e) INFLUENCE: drop each scored task from scoring in turn (router-minus-best-fixed over the remaining 10)")
    inf_rows, rows = [], []
    for t in sorted(scored, key=lambda x: x.tid):
        out = [f"{t.tid} {t.category[:24]}"]
        for lam in LAMBDA_MAIN:
            _, ag = evaluate(tasks, base_p, lam, scored_ids - {t.tid})
            out.append(f"{ag['diff']:+.3f}")
            inf_rows.append({"dropped_task": t.tid, "category": t.category, "lambda": lam,
                             "router_minus_best_fixed": round(ag["diff"], 6), "router": round(ag["router"], 6),
                             "best_fixed": ag["best_fixed"], "best_fixed_regret": round(ag["best_fixed_regret"], 6)})
        rows.append(out)
    for l in render(["dropped task"] + [f"lambda={l:g}" for l in LAMBDA_MAIN], rows):
        emit("    " + l)
    write_csv(os.path.join(OUT_DIR, "influence.csv"), inf_rows)

    cfg = {"estimator": "empirical-shrinkage", "k": k, "mc_draws": MC_DRAWS, "seed": SEED, "lambda_main": LAMBDA_MAIN,
           "lambda_grid": [round(x, 4) for x in LAMBDA_GRID], "metric": "structured (900001/900002); normalize_score elsewhere",
           "scored_tasks": sorted(scored_ids), "regret_excluded_ties": sorted(t.tid for t in ties)}
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    emit("\nFiles written under results/router/: " + ", ".join(sorted(os.listdir(OUT_DIR))))
    emit("and results/summary/: " + ", ".join(sorted(os.listdir(SUMMARY_DIR))))
    return P


# ============================================================================ pooled prior
class PooledPriorEstimator:
    """Pre-declared variant ("pooled prior"), added next to the zero-target estimator.

    WHY IT EXISTS: under the zero-target estimator the shrink factor n/(n+k) multiplies every
    architecture's advantage and cost deviation in a bucket by the same constant, so k can never
    change a pick -- a design flaw (k was a no-op). It was added to make the shrinkage parameter
    meaningful, NOT because of how either version performed. It is evaluated at k in {1, 2, 5}
    with no selection among them.

    DEFINITION: for each bucket and architecture
        value_hat = (n * bucket_value + k * pooled_value) / (n + k)
    applied to (a) the quality advantage and (b) the cost deviation (cost ratio - 1), with
        n            = distinct categories in the bucket (as in the zero-target version)
        bucket_value = mean over those categories of the category-mean value
        pooled_value = mean over ALL training categories (all buckets) of the category-mean value
                       (one vote per category; excludes the held-out category under LOCO, and
                       excludes only the held-out task under LOTO)
    An empty bucket (n = 0) therefore gets the pooled value, i.e. a prior-only pick (flagged
    "prior only (empty bucket)"), not an arbitrary default. cost_hat = C_bar * (1 + cost_dev_hat).

    CONFIDENCE: two-level bootstrap -- resample ALL training categories with replacement
    (5000 draws, fixed seed), recompute the pooled prior, the bucket's n and mean, shrink and
    decide -- so prior uncertainty is included. Reported only when the bucket has n >= 2
    distinct categories (same policy as the zero-target version).
    """
    name = "pooled-prior"
    default_flags = ("tie in estimate",)   # an empty bucket is resolved by the pooled prior

    def __init__(self, k=DEFAULT_K):
        self.k = k

    def fit(self, train, setting):
        self.c_bar = mean(mean(t.C.values()) for t in train)
        by_bucket_cat = defaultdict(lambda: defaultdict(list))
        by_cat = defaultdict(list)
        for t in train:
            qm, cm = mean(t.Q.values()), mean(t.C.values())
            adv = np.array([t.Q[a] - qm for a in ARCHS])
            dev = np.array([t.C[a] / cm - 1.0 for a in ARCHS])
            by_bucket_cat[t.bucket(setting)][t.category].append((adv, dev))
            by_cat[t.category].append((adv, dev))
        self.cat_names = sorted(by_cat)
        self.cat_adv = np.array([np.mean([x[0] for x in by_cat[c]], axis=0) for c in self.cat_names])
        self.cat_dev = np.array([np.mean([x[1] for x in by_cat[c]], axis=0) for c in self.cat_names])
        self.pool_adv = self.cat_adv.mean(axis=0)
        self.pool_dev = self.cat_dev.mean(axis=0)
        m = len(self.cat_names)
        self.buckets = {}
        for b, cats in by_bucket_cat.items():
            names = sorted(cats)
            advs = np.array([np.mean([x[0] for x in cats[c]], axis=0) for c in names])
            devs = np.array([np.mean([x[1] for x in cats[c]], axis=0) for c in names])
            has = np.zeros(m)
            badv, bdev = np.zeros((m, 3)), np.zeros((m, 3))
            for c in names:
                j = self.cat_names.index(c)
                has[j] = 1.0
                badv[j] = np.mean([x[0] for x in cats[c]], axis=0)
                bdev[j] = np.mean([x[1] for x in cats[c]], axis=0)
            self.buckets[b] = (names, advs, devs, has, badv, bdev)
        return self

    def estimate(self, bucket):
        if bucket not in self.buckets:
            dev = self.pool_dev.copy()
            return Estimate(0, [], self.pool_adv.copy(), self.c_bar * (1.0 + dev), np.zeros(3), None,
                            1.0 + dev, 0, True)
        names, advs, devs = self.buckets[bucket][:3]
        n = len(names)
        adv_hat = (n * advs.mean(axis=0) + self.k * self.pool_adv) / (n + self.k)
        dev_hat = (n * devs.mean(axis=0) + self.k * self.pool_dev) / (n + self.k)
        se = advs.std(axis=0, ddof=1) / math.sqrt(n) if n >= 2 else None
        n_inf = int((np.abs(advs).max(axis=1) > TIE_EPS).sum())
        return Estimate(n, names, adv_hat, self.c_bar * (1.0 + dev_hat), advs.mean(axis=0), se,
                        1.0 + dev_hat, n_inf, False, _effective_n(advs))

    def bootstrap(self, bucket, rng, draws=MC_DRAWS):
        if bucket not in self.buckets or len(self.buckets[bucket][0]) < 2:
            return None
        has, badv, bdev = self.buckets[bucket][3:]
        m = len(self.cat_names)
        idx = rng.integers(0, m, size=(draws, m))
        W = np.stack([(idx == j).sum(1) for j in range(m)], axis=1).astype(float)   # (draws, m)
        pool_adv = (W[:, :, None] * self.cat_adv[None]).sum(axis=1) / m
        pool_dev = (W[:, :, None] * self.cat_dev[None]).sum(axis=1) / m
        n_b = (W * has[None]).sum(axis=1)[:, None]
        sum_adv = (W[:, :, None] * badv[None]).sum(axis=1)
        sum_dev = (W[:, :, None] * bdev[None]).sum(axis=1)
        adv_hat = (sum_adv + self.k * pool_adv) / (n_b + self.k)
        dev_hat = (sum_dev + self.k * pool_dev) / (n_b + self.k)
        return adv_hat, self.c_bar * (1.0 + dev_hat)


VERSIONS = [  # (label, estimator class, k); labels appear in every comparison table
    ("zero-target", EmpiricalShrinkageEstimator, 2.0),
    ("pooled k=1", PooledPriorEstimator, 1.0),
    ("pooled k=2", PooledPriorEstimator, 2.0),
    ("pooled k=5", PooledPriorEstimator, 5.0),
]
POOLED_DIR = os.path.join(OUT_DIR, "pooled_prior")


def run_compare():
    """Zero-target vs pooled prior, side by side, every table labelled by version. Writes
    results/router/comparison_report.txt and new files under results/router/pooled_prior/;
    the zero-target outputs written by run() are not touched."""
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    tasks = load_tasks("structured")
    scored = [t for t in tasks if t.scored]
    scored_ids = {t.tid for t in scored}
    ties = sorted(t.tid for t in tasks if not t.scored)
    labs = [v[0] for v in VERSIONS]
    pooled_labs = [l for l in labs if l != "zero-target"]
    CAT = {t.tid: t.category for t in tasks}

    PV = {(lab, p, s): make_predictions(tasks, p, s, k, cls)
          for lab, cls, k in VERSIONS for p in PROTOCOLS for s in SETTINGS}
    EV = {(lab, p, s, lam): evaluate(tasks, PV[(lab, p, s)], lam)
          for lab in labs for p in PROTOCOLS for s in SETTINGS for lam in LAMBDA_MAIN}

    # the zero-target numbers recomputed here must equal the committed run() outputs
    with open(os.path.join(OUT_DIR, "summary_main.csv")) as f:
        base = {(r["protocol"], r["refusal_setting"], float(r["lambda"])): r for r in csv.DictReader(f)}
    for (lab, p, s, lam), (_, ag) in EV.items():
        if lab == "zero-target":
            r = base[(p, s, lam)]
            assert abs(ag["router"] - float(r["router_regret"])) < 1e-5 and \
                abs(ag["best_fixed_regret"] - float(r["best_fixed_regret"])) < 1e-5, (p, s, lam)

    def rec(ag, key="router"):
        return 1.0 - ag[key] / ag["best_fixed_regret"] if ag["best_fixed_regret"] > 0 else float("nan")

    emit("=" * 124)
    emit("COMPARISON: zero-target estimator vs pre-declared 'pooled prior' variant   (metric=structured; 11 scored tasks)")
    emit("=" * 124)
    emit("Labels: 'zero-target' = shrink toward 0 (k inert; k=2 shown).  'pooled k=K' = shrink toward the pooled all-category value,")
    emit("        (n*bucket + K*pooled)/(n+K). The variant was added because k was a no-op under the zero-target version (a design")
    emit("        flaw), not because of how either version performed; K in {1,2,5} are all reported, none selected.")
    emit("Refusal flag: 'true' = cached flag (upper bound, leaks the category label); 'llm' = flag as the LLM alone produced it.")
    emit("Regret* resolves default picks at random; 'default' = scored tasks picked by a tie or (zero-target only) an empty bucket;")
    emit("'prior' = scored tasks decided by the pooled prior alone (empty bucket, pooled variant). Zero-target numbers below were")
    emit("recomputed and asserted equal to the committed zero-target outputs, which are untouched.")

    # ---- 1. LOCO vs LOTO coincidence (zero-target)
    emit("\n1. DO THE TWO 0.120 VALUES COINCIDE?  (zero-target, refusal=true, lambda=0)")
    la, ag_a = EV[("zero-target", "LOCO", "true", 0.0)]
    lb, ag_b = EV[("zero-target", "LOTO", "true", 0.0)]
    ma, mb = {r["tid"]: r for r in la}, {r["tid"]: r for r in lb}
    rows = []
    for t in sorted(scored, key=lambda x: x.tid):
        a, b = ma[t.tid], mb[t.tid]
        rows.append([t.tid, t.category[:28], SHORT[a["oracle"]], SHORT[a["pick"]], a["flag"], f3(a["regret"]),
                     SHORT[b["pick"]], b["flag"], f3(b["regret"])])
    for l in render(["task", "category", "oracle", "LOCO pick", "LOCO basis", "LOCO regret", "LOTO pick", "LOTO basis",
                     "LOTO regret"], rows):
        emit("  " + l)
    same_all = sum(ma[t.tid]["pick"] == mb[t.tid]["pick"] for t in tasks)
    emit(f"  mean regret: LOCO {ag_a['router']:.4f}  LOTO {ag_b['router']:.4f};  picks identical on {same_all} of {len(tasks)} tasks "
         f"(all {len(scored)} scored).")
    emit(f"  They genuinely coincide. The picks are the same but not for the same reason: LOCO's picks on 36/89 are empty-bucket defaults")
    emit(f"  and on 1624/1654 an exact tie broken by cost; LOTO gets them from twin evidence. With defaults resolved at random LOCO is "
         f"{ag_a['router_tie']:.3f} vs LOTO {ag_b['router_tie']:.3f}.")

    # ---- 2. model card
    emit("\n2. WHAT EACH VERSION LEARNS (fit on all 24 tasks)")
    for setting in SETTINGS:
        emit(f"  refusal flag = {setting}")
        zt = EmpiricalShrinkageEstimator(2.0).fit(tasks, setting)
        pp = {k: PooledPriorEstimator(k).fit(tasks, setting) for k in (1.0, 2.0, 5.0)}
        emit("    pooled prior (mean over all %d categories): advantage SA/WF/MA = %s ; cost ratio SA/WF/MA = %s" %
             (len(pp[2.0].cat_names), "/".join(f"{x:+.3f}" for x in pp[2.0].pool_adv),
              "/".join(f"{1 + x:.2f}" for x in pp[2.0].pool_dev)))
        rows = []
        for b in sorted(zt.buckets):
            e0 = zt.estimate(b)
            row = [bucket_str(b), e0.n, "/".join(f"{x:+.3f}" for x in e0.adv_mean)]
            row.append("/".join(f"{x:+.3f}" for x in e0.adv_hat))
            for k in (1.0, 2.0, 5.0):
                row.append("/".join(f"{x:+.3f}" for x in pp[k].estimate(b).adv_hat))
            rows.append(row)
        for l in render(["bucket", "n cats", "bucket mean adv (unshrunk)", "zero-target adv_hat",
                         "pooled k=1 adv_hat", "pooled k=2 adv_hat", "pooled k=5 adv_hat"], rows):
            emit("    " + l)

    # ---- 3. headline tables
    emit("\n3. RESULTS ON THE 11 SCORED TASKS  (regret: lower is better; 'rec' = share of the best-fixed-vs-oracle headroom recovered, "
         "1 - router/best_fixed)")
    summary_rows = []
    for protocol in PROTOCOLS:
        for setting in SETTINGS:
            emit(f"\n  {protocol}{' (HEADLINE)' if protocol == 'LOCO' else ' (optimistic: twins stay in training)'}, refusal={setting}"
                 + (" [upper bound]" if setting == "true" else " [LLM-only flag]"))
            rows = []
            for lam in LAMBDA_MAIN:
                for lab in labs:
                    _, ag = EV[(lab, protocol, setting, lam)]
                    rows.append([f"{lam:g}", lab, f3(ag["router"]), f3(ag["router_tie"]), f"{ag['agree']}/{ag['n_scored']}",
                                 ag["n_default"], ag["n_prior_only"], f"{SHORT[ag['best_fixed']]} {f3(ag['best_fixed_regret'])}",
                                 f"{ag['diff']:+.3f}", f"{ag['diff_tie']:+.3f}", f"{rec(ag)*100:.0f}%", f"{rec(ag, 'router_tie')*100:.0f}%"])
                    summary_rows.append({
                        "estimator": lab, "protocol": protocol, "refusal_setting": setting, "lambda": lam,
                        "router_regret": round(ag["router"], 6), "router_regret_ties_averaged": round(ag["router_tie"], 6),
                        "oracle_agreement": ag["agree"], "n_scored": ag["n_scored"], "n_default_picks": ag["n_default"],
                        "n_prior_only_picks": ag["n_prior_only"], "best_fixed": ag["best_fixed"],
                        "best_fixed_regret": round(ag["best_fixed_regret"], 6),
                        "router_minus_best_fixed": round(ag["diff"], 6),
                        "router_minus_best_fixed_ties_averaged": round(ag["diff_tie"], 6),
                        "headroom_recovered": round(rec(ag), 4), "headroom_recovered_ties_averaged": round(rec(ag, "router_tie"), 4)})
                rows.append([""] * 12)
            for l in render(["lambda", "version", "router", "router*", "agree", "default", "prior", "best fixed", "router-best",
                             "router*-best", "rec", "rec*"], rows[:-1]):
                emit("    " + l)
    write_csv(os.path.join(POOLED_DIR, "summary_main.csv"), summary_rows)

    # ---- 4. headroom recovered
    emit("\n4. SHARE OF THE BEST-FIXED-VS-ORACLE HEADROOM RECOVERED  (1 - router regret / best-fixed regret; ties-averaged in brackets; "
         "negative = worse than the best fixed strategy)")
    cols = [(p, s) for p in PROTOCOLS for s in SETTINGS]
    hd_rows = []
    for lam in LAMBDA_MAIN:
        emit(f"\n  lambda={lam:g}   (best fixed: " +
             ", ".join(f"{p}/{s} {SHORT[EV[('zero-target', p, s, lam)][1]['best_fixed']]} "
                       f"{EV[('zero-target', p, s, lam)][1]['best_fixed_regret']:.3f}" for p, s in cols[:2]) + ")")
        rows = []
        for lab in labs:
            row = [lab]
            for p, s in cols:
                _, ag = EV[(lab, p, s, lam)]
                row.append(f"{rec(ag)*100:+.0f}% ({rec(ag, 'router_tie')*100:+.0f}%)")
                hd_rows.append({"estimator": lab, "protocol": p, "refusal_setting": s, "lambda": lam,
                                "headroom_recovered": round(rec(ag), 4), "headroom_recovered_ties_averaged": round(rec(ag, "router_tie"), 4)})
            rows.append(row)
        for l in render(["version"] + [f"{p} {s}" for p, s in cols], rows):
            emit("    " + l)
    write_csv(os.path.join(POOLED_DIR, "headroom.csv"), hd_rows)

    # ---- 5. picks that differ from zero-target
    emit("\n5. PICKS THAT DIFFER FROM THE ZERO-TARGET VERSION  (task: zero-target -> pooled; scored tasks starred)")
    diff_rows = []
    for protocol in PROTOCOLS:
        for setting in SETTINGS:
            for lab in pooled_labs:
                for lam in LAMBDA_MAIN:
                    zr = {r["tid"]: r for r in EV[("zero-target", protocol, setting, lam)][0]}
                    pr = {r["tid"]: r for r in EV[(lab, protocol, setting, lam)][0]}
                    for tid in sorted(zr):
                        if zr[tid]["pick"] != pr[tid]["pick"]:
                            diff_rows.append({"estimator": lab, "protocol": protocol, "refusal_setting": setting, "lambda": lam,
                                              "task_id": tid, "scored": tid in scored_ids, "zero_target_pick": zr[tid]["pick"],
                                              "pooled_pick": pr[tid]["pick"], "oracle_pick": zr[tid]["oracle"]})
    write_csv(os.path.join(POOLED_DIR, "picks_diff.csv"), diff_rows or [{"note": "no differences"}])
    for protocol in PROTOCOLS:
        for setting in SETTINGS:
            emit(f"\n  {protocol}, refusal={setting}")
            rows = []
            for lab in pooled_labs:
                for lam in (0.0, 100.0):
                    zr = {r["tid"]: r for r in EV[("zero-target", protocol, setting, lam)][0]}
                    pr = {r["tid"]: r for r in EV[(lab, protocol, setting, lam)][0]}
                    d_sc = [f"*{tid}: {SHORT[zr[tid]['pick']]}->{SHORT[pr[tid]['pick']]}" for tid in sorted(zr)
                            if tid in scored_ids and zr[tid]["pick"] != pr[tid]["pick"]]
                    d_un = sum(1 for tid in zr if tid not in scored_ids and zr[tid]["pick"] != pr[tid]["pick"])
                    rows.append([lab, f"{lam:g}", ", ".join(d_sc) if d_sc else "none", d_un])
            for l in render(["version", "lambda", "scored tasks whose pick differs", "# unscored (tied) tasks differing"], rows):
                emit("    " + l)
    emit("\n  36, 89, 1624, 1654 explicitly (pick / oracle at lambda=0; evidence basis in brackets):")
    for protocol in PROTOCOLS:
        for setting in SETTINGS:
            rows = []
            for lab in labs:
                res = {r["tid"]: r for r in EV[(lab, protocol, setting, 0.0)][0]}
                rows.append([lab] + [f"{SHORT[res[t]['pick']]}/{SHORT[res[t]['oracle']]} [{res[t]['flag'].split(' (')[0]}]"
                                     for t in (36, 89, 1624, 1654)])
            emit(f"    {protocol}, refusal={setting}")
            for l in render(["version", "36", "89", "1624", "1654"], rows):
                emit("      " + l)

    # ---- 6. width-2
    emit("\n6. WIDTH-2 TASKS (900001, 900002) UNDER LEAVE-ONE-CATEGORY-OUT")
    rows = []
    for setting in SETTINGS:
        for lam in LAMBDA_MAIN:
            for lab in labs:
                res, _ = EV[(lab, "LOCO", setting, lam)]
                rm = {r["tid"]: r for r in res}
                tot = sum(r["regret"] for r in res if r["tid"] in scored_ids)
                for tid in (900001, 900002):
                    r = rm[tid]
                    rows.append([setting, f"{lam:g}", lab, tid, f"{SHORT[r['pick']]} ({r['flag'].split(' (')[0]})", SHORT[r["oracle"]],
                                 f3(r["regret"]), pct(r["regret"] / tot if tot > 0 else 0.0)])
    for l in render(["flag", "lambda", "version", "task", "pick (basis)", "oracle", "regret", "share of total regret"], rows):
        emit("  " + l)

    # ---- 7. lambda grid
    emit("\n7. LAMBDA GRID (saved to results/router/pooled_prior/lambda_sweep.csv and lambda_curve.csv); router-minus-best-fixed, "
         "LOCO, refusal=true")
    sweep_rows, curve_rows = [], {}
    for lab in labs:
        for protocol, setting in (("LOCO", "true"), ("LOCO", "llm"), ("LOTO", "true"), ("LOTO", "llm")):
            for lam in LAMBDA_GRID:
                res, ag = evaluate(tasks, PV[(lab, protocol, setting)], lam)
                curve_rows[(lab, protocol, setting, lam)] = ag
                if protocol == "LOCO":
                    for r in res:
                        sweep_rows.append({"estimator": lab, "refusal_setting": setting, "lambda": round(lam, 4), "task_id": r["tid"],
                                           "scored": r["scored"], "router_pick": r["pick"], "evidence": r["flag"],
                                           "oracle_pick": r["oracle"], "regret": round(r["regret"], 6),
                                           "U_single_agent": round(r["U"]["single_agent"], 6), "U_workflow": round(r["U"]["workflow"], 6),
                                           "U_multi_agent": round(r["U"]["multi_agent"], 6)})
    write_csv(os.path.join(POOLED_DIR, "lambda_sweep.csv"), sweep_rows)
    write_csv(os.path.join(POOLED_DIR, "lambda_curve.csv"),
              [{"estimator": lab, "protocol": p, "refusal_setting": s, "lambda": round(lam, 4), "router": round(ag["router"], 6),
                "router_ties_averaged": round(ag["router_tie"], 6), "best_fixed": ag["best_fixed"],
                "best_fixed_regret": round(ag["best_fixed_regret"], 6), "router_minus_best_fixed": round(ag["diff"], 6),
                "headroom_recovered": round(rec(ag), 4)} for (lab, p, s, lam), ag in curve_rows.items()])
    sel = [LAMBDA_GRID[i] for i in (0, 1, 8, 14, 17, 22, 25)]
    rows = []
    for lam in sel:
        row = [f"{lam:g}"]
        for lab in labs:
            ag = curve_rows[(lab, "LOCO", "true", lam)]
            row.append(f"{ag['diff']:+.3f} ({ag['router']:.3f} vs {SHORT[ag['best_fixed']]} {ag['best_fixed_regret']:.3f})")
        rows.append(row)
    for l in render(["lambda"] + labs, rows):
        emit("  " + l)

    # ---- 8. bootstrap
    emit("\n8. UNCERTAINTY: bootstrap over categories, 95% CI of (router regret - best fixed regret), LOCO   [negative = router better]")
    ci_rows, rows = [], []
    for setting in SETTINGS:
        for lam in LAMBDA_MAIN:
            for lab in labs:
                res, ag = EV[(lab, "LOCO", setting, lam)]
                ci = bootstrap_ci(tasks, res, lam, scored_ids)
                ci_t = bootstrap_ci(tasks, res, lam, scored_ids, use_tie=True)
                rows.append([setting, f"{lam:g}", lab, f"{ag['diff']:+.3f}", f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]",
                             f"{ci['p_router_better']*100:.0f}%", f"[{ci_t['lo']:+.3f}, {ci_t['hi']:+.3f}]"])
                ci_rows.append({"estimator": lab, "refusal_setting": setting, "lambda": lam, "router_minus_best_fixed": round(ag["diff"], 6),
                                "ci_lo": round(ci["lo"], 6), "ci_hi": round(ci["hi"], 6), "p_router_better": round(ci["p_router_better"], 4),
                                "ci_lo_ties_averaged": round(ci_t["lo"], 6), "ci_hi_ties_averaged": round(ci_t["hi"], 6)})
    for l in render(["flag", "lambda", "version", "router-best", "95% CI", "P(router better)", "95% CI (ties averaged)"], rows):
        emit("  " + l)
    write_csv(os.path.join(POOLED_DIR, "bootstrap_ci.csv"), ci_rows)

    # ---- 9. sensitivity + influence
    emit("\n9. SENSITIVITY (LOCO, refusal=true; router-minus-best-fixed regret, router vs best fixed in brackets). The (c) k-sweep is the set "
         "of 'pooled k=1/2/5' rows; the zero-target row is identical for every k.")
    tasks_f1 = load_tasks("f1")
    scored_f1 = {t.tid for t in tasks_f1 if t.scored}
    P_f1 = {lab: make_predictions(tasks_f1, "LOCO", "true", k, cls) for lab, cls, k in VERSIONS}
    variants = [("baseline", tasks, scored_ids, None),
                ("(a) drop 722 from scoring", tasks, scored_ids - {722}, None),
                ("(b) drop 900001+900002 from scoring", tasks, scored_ids - {900001, 900002}, None),
                ("(d) F1 metric instead of structured", tasks_f1, scored_f1, P_f1)]
    sens_rows, rows = [], []
    for vname, tv, sv, pf in variants:
        for lab in labs:
            preds = PV[(lab, "LOCO", "true")] if pf is None else pf[lab]
            out = [vname, lab]
            for lam in LAMBDA_MAIN:
                _, ag = evaluate(tv, preds, lam, sv)
                out.append(f"{ag['diff']:+.3f} ({ag['router']:.3f} vs {SHORT[ag['best_fixed']]} {ag['best_fixed_regret']:.3f})")
                sens_rows.append({"variant": vname, "estimator": lab, "lambda": lam, "router": round(ag["router"], 6),
                                  "best_fixed": ag["best_fixed"], "best_fixed_regret": round(ag["best_fixed_regret"], 6),
                                  "router_minus_best_fixed": round(ag["diff"], 6), "n_scored": ag["n_scored"]})
            rows.append(out)
    for l in render(["variant", "version"] + [f"lambda={l:g}" for l in LAMBDA_MAIN], rows):
        emit("  " + l)
    write_csv(os.path.join(POOLED_DIR, "sensitivity.csv"), sens_rows)

    emit("\n   (e) INFLUENCE: drop each scored task from scoring in turn (router-minus-best-fixed over the remaining 10; LOCO, refusal=true)")
    inf_rows, rows = [], []
    for t in sorted(scored, key=lambda x: x.tid):
        out = [f"{t.tid} {t.category[:22]}"]
        for lab in labs:
            for lam in (0.0, 100.0):
                _, ag = evaluate(tasks, PV[(lab, "LOCO", "true")], lam, scored_ids - {t.tid})
                out.append(f"{ag['diff']:+.3f}")
            for lam in LAMBDA_MAIN:
                _, ag = evaluate(tasks, PV[(lab, "LOCO", "true")], lam, scored_ids - {t.tid})
                inf_rows.append({"dropped_task": t.tid, "estimator": lab, "lambda": lam, "router_minus_best_fixed": round(ag["diff"], 6),
                                 "router": round(ag["router"], 6), "best_fixed": ag["best_fixed"],
                                 "best_fixed_regret": round(ag["best_fixed_regret"], 6)})
        rows.append(out)
    hdr = ["dropped task"] + [f"{lab} @{lam:g}" for lab in labs for lam in (0.0, 100.0)]
    for l in render(hdr, rows):
        emit("  " + l)
    write_csv(os.path.join(POOLED_DIR, "influence.csv"), inf_rows)

    # ---- 10. confidence diagnostics
    emit("\n10. WHEN IS THE MONTE CARLO CONFIDENCE NOT TRUSTWORTHY?  (task 722's 100% for single-agent, and every comparable case)")
    est722 = EmpiricalShrinkageEstimator(2.0).fit([x for x in tasks if x.category != "knowledge_qa"], "true")
    names, advs, ratios = est722.buckets[(1, False)]
    e722 = est722.estimate((1, False))
    inf_idx = [i for i in range(len(names)) if np.abs(advs[i]).max() > TIE_EPS]
    r722 = {r["tid"]: r for r in EV[("zero-target", "LOCO", "true", 0.0)][0]}[722]
    emit("  Task 722 (knowledge_qa), zero-target, LOCO, refusal=true, lambda=0:")
    emit(f"    bucket (w1, no-refusal) has n={e722.n} categories once knowledge_qa is held out; the router's pick is single_agent.")
    emit(f"    Only {len(inf_idx)} of them have ANY quality difference between architectures (the other {e722.n - len(inf_idx)} are exact zeros);")
    emit(f"    effective number of categories carrying the evidence (participation ratio) = {e722.eff_informative:.2f}:")
    for i in inf_idx:
        emit(f"      {names[i]:34} advantage SA/WF/MA = " + "/".join(f"{x:+.3f}" for x in advs[i]))
    emit(f"    bucket mean advantage SA/WF/MA = " + "/".join(f"{x:+.4f}" for x in e722.adv_mean) +
         "   SE across categories = " + "/".join(f"{x:.4f}" for x in e722.adv_se))
    gap = e722.adv_mean[0] - max(e722.adv_mean[1], e722.adv_mean[2])
    emit(f"    SA's lead over the next architecture = {gap:.4f}  ({gap / max(e722.adv_se[0], 1e-12):.1f} x SA's own SE; that SE comes from just those {len(inf_idx)} categories)")
    p_miss = ((e722.n - len(inf_idx)) / e722.n) ** e722.n
    emit(f"    cost term: at lambda=0 it is not used at all (cost_hat only breaks exact ties); at lambda>0 SA is also the cheapest estimated "
         f"(cost ratios SA/WF/MA = " + "/".join(f"{x:.2f}" for x in e722.rho_hat) + "), which only reinforces it.")
    emit(f"    bootstrap: a resample of the {e722.n} categories contains neither informative category with probability "
         f"({e722.n - len(inf_idx)}/{e722.n})^{e722.n} = {p_miss*100:.1f}%.  In such a draw every architecture's advantage is exactly 0, the top score")
    emit(f"    is an exact tie, and argmax returns the FIRST index = single_agent. Measured share of exact-tie draws: {r722['tie_draw_share']*100:.1f}%.")
    emit(f"    Reported confidence (ties -> index 0): {r722['confidence']*100:.1f}%.   With ties resolved by the point-decision rule (lowest cost, then priority): "
         f"{r722['confidence_tiebreak']*100:.1f}%.")
    emit(f"    So the 100% is ~{(1 - r722['tie_draw_share'])*100:.0f}% 'at least one informative category was resampled' plus ~{r722['tie_draw_share']*100:.0f}% 'nothing to compare, tie "
         f"went to single_agent by index order'. It measures how often the bucket MEAN ordering survives resampling 2 informative categories,")
    emit(f"    not whether single_agent is right for 722: 722's own oracle is workflow (by 0.5 points, run-to-run noise).")
    emit("")
    emit("  Cases where a confidence figure should not be trusted (rule-based, applied to every confidence the router reports):")
    emit("    (1) NOT REPORTED, by design: bucket n <= 1 distinct categories (one category gives no spread), an empty bucket, or an exact tie in the point estimate.")
    emit("    (2) n = 2: reported but weak (a bootstrap of two categories has only three distinct resamples).")
    emit("    (3) TIE-INFLATED: >= 5% of draws are exact ties that the bootstrap resolves by index order (single_agent), unlike the point rule.")
    emit("    (4) THIN: the evidence rests on fewer than 3 EFFECTIVE categories (participation ratio of category influence, which")
    emit("        counts two categories of very different size as ~1.8; a category with a tiny advantage barely adds to it).")
    emit("    (5) always: it is a statement about the BUCKET's mean ordering, not a calibrated probability that the pick is right for the task")
    emit("        (nothing here is calibrated: 11 scored tasks, and the confidence ignores run-to-run noise inside a category).")
    diag_rows = []
    for lab in labs:
        for protocol in PROTOCOLS:
            for setting in SETTINGS:
                for lam in LAMBDA_MAIN:
                    for r in EV[(lab, protocol, setting, lam)][0]:
                        na = r["confidence"] is None
                        flags = []
                        if not na:
                            if r["n"] == 2:
                                flags.append("n=2 weak")
                            if r["tie_draw_share"] >= 0.05:
                                flags.append("tie-inflated")
                            if r["eff_informative"] < 3.0:
                                flags.append("thin (<3 effective categories)")
                        diag_rows.append({"estimator": lab, "protocol": protocol, "refusal_setting": setting, "lambda": lam,
                                          "task_id": r["tid"], "scored": r["scored"], "n_categories": r["n"],
                                          "n_informative_categories": r["n_informative"],
                                          "effective_informative_categories": round(r["eff_informative"], 3),
                                          "pick": r["pick"], "evidence": r["flag"],
                                          "confidence": "" if na else round(r["confidence"], 4),
                                          "tie_draw_share": "" if na else round(r["tie_draw_share"], 4),
                                          "confidence_ties_by_point_rule": "" if na else round(r["confidence_tiebreak"], 4),
                                          "not_trustworthy_flags": "not reported" if na else ";".join(flags)})
    write_csv(os.path.join(POOLED_DIR, "confidence_diagnostics.csv"), diag_rows)
    emit("")
    emit("  Counts over every (version, protocol, flag, lambda) x 24 tasks = %d router decisions:" % len(diag_rows))
    rows = []
    for lab in labs:
        sub = [d for d in diag_rows if d["estimator"] == lab]
        rep = [d for d in sub if d["confidence"] != ""]
        rows.append([lab, len(sub), len(sub) - len(rep), len(rep), sum("n=2 weak" in d["not_trustworthy_flags"] for d in rep),
                     sum("tie-inflated" in d["not_trustworthy_flags"] for d in rep), sum("thin" in d["not_trustworthy_flags"] for d in rep),
                     sum(d["not_trustworthy_flags"] == "" for d in rep)])
    for l in render(["version", "decisions", "no confidence", "confidence shown", "n=2 weak", "tie-inflated", "thin", "shown with none of these flags"], rows):
        emit("    " + l)
    emit("")
    emit("  Zero-target, LOCO, refusal=true, lambda=0: every decision that shows a confidence")
    rows = []
    for r in sorted(EV[("zero-target", "LOCO", "true", 0.0)][0], key=lambda x: (not x["scored"], x["tid"])):
        if r["confidence"] is None:
            continue
        fl = [d["not_trustworthy_flags"] for d in diag_rows if d["estimator"] == "zero-target" and d["protocol"] == "LOCO"
              and d["refusal_setting"] == "true" and d["lambda"] == 0.0 and d["task_id"] == r["tid"]][0]
        rows.append([("*" if r["scored"] else " ") + str(r["tid"]), SHORT[r["pick"]], r["n"], r["n_informative"],
                     f"{r['eff_informative']:.2f}", f"{r['confidence']*100:.0f}%",
                     f"{r['tie_draw_share']*100:.0f}%", f"{r['confidence_tiebreak']*100:.0f}%", fl or "-"])
    for l in render(["task", "pick", "n cats", "any-diff cats", "effective cats", "confidence", "tie draws",
                     "conf (ties by point rule)", "flags"], rows):
        emit("    " + l)
    emit("  Pooled variants: the two-level bootstrap also resamples the pooled prior, so confidences shown for pooled rows include prior")
    emit("  uncertainty; the same flags apply (see confidence_diagnostics.csv for every row).")

    cfg = {"versions": [{"label": l, "estimator": c.name, "k": k} for l, c, k in VERSIONS], "mc_draws": MC_DRAWS, "seed": SEED,
           "lambda_main": LAMBDA_MAIN, "lambda_grid": [round(x, 4) for x in LAMBDA_GRID], "metric": "structured",
           "scored_tasks": sorted(scored_ids), "regret_excluded_ties": ties}
    os.makedirs(POOLED_DIR, exist_ok=True)
    with open(os.path.join(POOLED_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(OUT_DIR, "comparison_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    emit("\nNew files: results/router/comparison_report.txt and results/router/pooled_prior/: " + ", ".join(sorted(os.listdir(POOLED_DIR))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--metric", choices=oracle.METRICS, default="structured")
    r = sub.add_parser("run")
    r.add_argument("--k", type=float, default=DEFAULT_K)
    sub.add_parser("compare")
    args = ap.parse_args()
    if args.cmd == "export":
        p, rows = export_quality_table(args.metric)
        print(f"wrote {os.path.relpath(p, REPO)} ({len(rows)} rows, metric={args.metric})")
    elif args.cmd == "compare":
        run_compare()
    else:
        run(args.k)
