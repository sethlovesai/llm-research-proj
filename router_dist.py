"""
router_dist.py -- bucket-based SOFT-COUNT Beta-posterior router, next to router.py.

Implements results/router_dist/SPEC.md exactly (pre-registered and committed before this file
existed). router.py / oracle.py are imported and reused (loading, oracle pick, regret aggregation,
category bootstrap) and are NOT modified.

Model (soft-count Beta update; NOT an exact Beta-binomial likelihood):
  evidence unit = one task category; its mean quality in [0,1] is a fractional success count.
  bucket b, architecture k:  n_b = #categories, s_bk = sum of category-mean qualities,
  posterior Beta(1 + s_bk, 1 + n_b - s_bk), mu = (1+s)/(2+n), sd = sqrt(mu(1-mu)/(n+3)).
  utility U_k = mu_k - lambda * c_hat_k; deterministic argmax, ties by lower cost, then SA<WF<MA.

Headline bucket definition: WIDTH ONLY. (width, depth) and the two privacy-flag definitions are
labelled comparison / exploratory variants.

Run:  ./venv/bin/python3 router_dist.py
Writes only under results/router_dist/.
"""
import glob
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

import oracle
import router as R

ARCHS, SHORT, PRIORITY = R.ARCHS, R.SHORT, R.PRIORITY
OUT = os.path.join(R.REPO, "results", "router_dist")
LAMBDAS = R.LAMBDA_MAIN
DRAWS, SEED, TIE_EPS = R.MC_DRAWS, R.SEED, R.TIE_EPS
PROTECTED = ["router.py", "oracle.py", "structured_score.py", "normalize_score.py", "decomposition"]
EXISTING_RESULT_DIRS = ["custom_baseline", "workflow", "multi_agent"]
PRIV_TASKS = {1624, 1654, 1914, 1936}

# bucket definitions: label -> key function over a task; HEADLINE first
DEFS = {
    "width (HEADLINE)": lambda t: (t.width,),
    "width+depth (exploratory)": lambda t: (t.width, t.depth),
    "width+privacy_category_flag (diagnostic; leaks category)": lambda t: (t.width, t.refusal_true),
    "width+privacy_LLM_flag (diagnostic)": lambda t: (t.width, t.refusal_llm),
}
HEAD = "width (HEADLINE)"
SIMPLE = {HEAD: "width", "width+depth (exploratory)": "width+depth",
          "width+privacy_category_flag (diagnostic; leaks category)": "width+priv_cat",
          "width+privacy_LLM_flag (diagnostic)": "width+priv_llm"}


def key_str(label, key):
    if label.startswith("width+depth"):
        return f"(w{key[0]}, d{key[1]})"
    if label.startswith("width+privacy"):
        return f"(w{key[0]}, {'refusal' if key[1] else 'no-refusal'})"
    return f"(w{key[0]})"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


# ------------------------------------------------------------------------------ data
def load(metric):
    """router.load_tasks plus depth and per-run quality/cost lists (run-level sensitivity)."""
    tasks = R.load_tasks(metric)
    for t in tasks:
        with open(os.path.join(oracle.CACHE_DIR, f"{t.tid}.json")) as f:
            d = json.load(f)
        feats = d.get("features") or {}
        t.depth = int(feats.get("critical_path_depth") or 1) if d["decomposition_warranted"] else 1
        t.runQ, t.runC = {}, {}
        for a in ARCHS:
            runs = [r for r in oracle._load_runs(t.tid, a) if not r["infra_error"]]
            t.runQ[a] = [oracle._run_quality(r["result"], t.tid, metric) for r in runs]
            t.runC[a] = [r["result"]["agent_info"]["total_cost"] for r in runs]
            assert abs(mean(t.runQ[a]) - t.Q[a]) < 1e-9 and abs(mean(t.runC[a]) - t.C[a]) < 1e-12
    return tasks


# ------------------------------------------------------------------------------ model
@dataclass
class Bucket:
    n: int
    s: np.ndarray
    chat: np.ndarray
    se: object          # None when unavailable (n == 1)
    chat_obs: np.ndarray

    @property
    def alpha(self):
        return 1.0 + self.s

    @property
    def beta(self):
        return 1.0 + self.n - self.s

    @property
    def mu(self):
        return (1.0 + self.s) / (2.0 + self.n)

    @property
    def sd(self):
        return np.sqrt(self.mu * (1 - self.mu) / (self.n + 3.0))


class BetaBucketModel:
    """unit: 'category' (headline) or 'run' (overconfident sensitivity d).
    cost_mode: 'observed' (headline) or 'ratio' (sensitivity e: router.py's ratio-anchored estimator)."""

    def __init__(self, keyfn, unit="category", cost_mode="observed", k_ratio=2.0):
        self.keyfn, self.unit, self.cost_mode, self.k_ratio = keyfn, unit, cost_mode, k_ratio

    def fit(self, train):
        # same global cost fallback as router.EmpiricalShrinkageEstimator.fit
        self.c_bar = mean(mean(t.C.values()) for t in train)
        groups = defaultdict(lambda: defaultdict(list))
        for t in train:
            groups[self.keyfn(t)][t.category].append(t)
        self.buckets = {}
        for key, cats in groups.items():
            if self.unit == "category":
                qs, cs, rat = [], [], []
                for c, ts in sorted(cats.items()):
                    # evidence unit: category mean over tasks and repeats (equal repeats asserted)
                    assert len({t.n_runs[a] for t in ts for a in ARCHS}) == 1
                    qs.append([mean(t.Q[a] for t in ts) for a in ARCHS])
                    cs.append([mean(t.C[a] for t in ts) for a in ARCHS])
                    rat.append([mean(t.C[a] / mean(t.C.values()) for t in ts) for a in ARCHS])
                qs, cs, rat = np.array(qs), np.array(cs), np.array(rat)
                n = len(qs)
            else:  # run-level units: every run of every training task is one unit
                ts = [t for cs_ in cats.values() for t in cs_]
                n = len(ts[0].runQ[ARCHS[0]]) * len(ts)
                qs = np.array([[q for t in ts for q in t.runQ[a]] for a in ARCHS]).T
                cs = np.array([[c for t in ts for c in t.runC[a]] for a in ARCHS]).T
                rat = np.array([[t.runC[a][i] / mean(t.runC[b][i] for b in ARCHS) for t in ts for i in range(len(t.runC[a]))]
                                for a in ARCHS]).T
                assert qs.shape[0] == n
            chat_obs = cs.mean(0)
            se = cs.std(axis=0, ddof=1) / math.sqrt(n) if n >= 2 else None
            if self.cost_mode == "ratio":
                w = n / (n + self.k_ratio)
                chat = self.c_bar * (1.0 + w * (rat.mean(0) - 1.0))
            else:
                chat = chat_obs
            self.buckets[key] = Bucket(n, qs.sum(0), chat, se, chat_obs)
        return self

    def posterior(self, key):
        if key in self.buckets:
            return self.buckets[key]
        return Bucket(0, np.zeros(3), self.c_bar * np.ones(3), None, self.c_bar * np.ones(3))


def decide(mu, chat, lam):
    U = mu - lam * chat
    tied = [i for i in range(3) if U.max() - U[i] < TIE_EPS]
    pick = sorted(tied, key=lambda i: (chat[i], i))[0]
    return ARCHS[pick], [ARCHS[i] for i in tied], U


def simulate(b, tid):
    """Q ~ Beta posterior, C ~ Normal(c_hat, SE) clipped at 0 (fixed when SE unavailable)."""
    rng = np.random.default_rng([SEED, tid])
    q = rng.beta(b.alpha, b.beta, size=(DRAWS, 3))
    z = rng.standard_normal((DRAWS, 3))
    se = np.zeros(3) if b.se is None else b.se
    c = np.clip(b.chat + z * se, 0.0, None)
    return q, c


def task_record(t, b, key_label, lam, qc):
    pick, tied, U = decide(b.mu, b.chat, lam)
    op = R.oracle_pick(t, lam)
    ou = R.actual_U(t, op, lam)
    q, c = qc
    Ud = q - lam * c
    # best architecture per draw (continuous draws; ties negligible; deterministic fallback by cost, index)
    best = Ud.argmax(axis=1)
    p_best = float((best == ARCHS.index(pick)).mean())
    pi = ARCHS.index(pick)
    others = np.delete(Ud, pi, axis=1).max(axis=1)
    gap = Ud[:, pi] - others
    lo, hi = float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5))
    if b.n == 0:
        flag = "no evidence (empty bucket)"
    elif len(tied) > 1:
        flag = "tie in estimate"
    elif b.n == 1:
        flag = "n=1 (cost SE unavailable)"
    else:
        flag = "ok"
    return {
        "tid": t.tid, "category": t.category, "scored": t.scored, "width": t.width, "depth": t.depth,
        "bucket": key_label, "n_cat": b.n, "pick": pick, "tied": tied, "oracle": op, "flag": flag,
        "is_default": flag in ("no evidence (empty bucket)", "tie in estimate"),
        "P_best": p_best, "gap_lo": lo, "gap_hi": hi, "no_clear_winner": lo <= 0.0 <= hi,
        "mu": b.mu.copy(), "sd": b.sd.copy(), "chat": b.chat.copy(),
        "regret": ou - R.actual_U(t, pick, lam),
        "regret_tie": mean(ou - R.actual_U(t, a, lam) for a in tied),
        "agree": pick == op, "agree_tie": float(op in tied) / len(tied),
    }


def cross_validate(tasks, label, protocol, unit="category", cost_mode="observed"):
    """records[lam][tid]. Fits one model per held-out category (LOCO) / task (LOTO)."""
    keyfn = DEFS[label]
    recs = {lam: {} for lam in LAMBDAS}
    folds = 0
    cache = {}
    for t in tasks:
        gk = t.category if protocol == "LOCO" else t.tid
        if gk not in cache:
            if protocol == "LOCO":
                train = [x for x in tasks if x.category != t.category]
                assert all(x.category != t.category for x in train)
            else:
                train = [x for x in tasks if x.tid != t.tid]
                assert all(x.tid != t.tid for x in train)
            cache[gk] = BetaBucketModel(keyfn, unit, cost_mode).fit(train)
            folds += 1
        m = cache[gk]
        key = keyfn(t)
        b = m.posterior(key)
        qc = simulate(b, t.tid)
        for lam in LAMBDAS:
            recs[lam][t.tid] = task_record(t, b, key_str(label, key), lam, qc)
    return recs, folds


def summarize(tasks, recs, lam, scored_ids):
    results = list(recs[lam].values())
    agg = R.aggregate(results, tasks, lam, scored_ids)
    ci = R.bootstrap_ci(tasks, results, lam, scored_ids)
    ci_t = R.bootstrap_ci(tasks, results, lam, scored_ids, use_tie=True)
    bf = agg["best_fixed_regret"]
    agg.update({
        "ci": ci, "ci_tie": ci_t,
        "headroom": bf,
        "recovered": (1 - agg["router"] / bf) if bf > 0 else float("nan"),
        "recovered_tie": (1 - agg["router_tie"] / bf) if bf > 0 else float("nan"),
    })
    return agg


def scored_set(tasks, drop=()):
    return {t.tid for t in tasks if t.scored} - set(drop)


# ---------------------------------------------------------------------------- reporting
def pct(x):
    return "n/a" if x != x else f"{x * 100:+.0f}%"


def summary_rows(tasks, recs, scored, tag):
    rows, csv = [], []
    for lam in LAMBDAS:
        a = summarize(tasks, recs, lam, scored)
        rows.append([f"{lam:g}", f"{R.f3(a['router_tie'])} ({R.f3(a['router'])})", a["n_default"], f"{a['agree']}/{a['n_scored']}",
                     f"{a['agree_tie']:.1f}", f"{SHORT[a['best_fixed']]} {R.f3(a['best_fixed_regret'])}",
                     f"{R.f3(a['fixed']['single_agent'])}/{R.f3(a['fixed']['workflow'])}/{R.f3(a['fixed']['multi_agent'])}",
                     f"{a['diff_tie']:+.3f} ({a['diff']:+.3f})",
                     f"[{a['ci_tie']['lo']:+.3f}, {a['ci_tie']['hi']:+.3f}] ([{a['ci']['lo']:+.3f}, {a['ci']['hi']:+.3f}])",
                     f"{pct(a['recovered_tie'])} ({pct(a['recovered'])})"])
        csv.append({"set": tag, "lambda": lam, "n_scored": a["n_scored"], "router_regret_tie_averaged": round(a["router_tie"], 6),
                    "router_regret_deterministic": round(a["router"], 6), "n_default_picks": a["n_default"],
                    "oracle_agree": a["agree"], "oracle_agree_tie_averaged": round(a["agree_tie"], 4),
                    "best_fixed": a["best_fixed"], "best_fixed_regret": round(a["best_fixed_regret"], 6),
                    "always_SA": round(a["fixed"]["single_agent"], 6), "always_WF": round(a["fixed"]["workflow"], 6),
                    "always_MA": round(a["fixed"]["multi_agent"], 6), "router_minus_best_fixed_tie": round(a["diff_tie"], 6),
                    "router_minus_best_fixed_det": round(a["diff"], 6), "ci_lo_tie": round(a["ci_tie"]["lo"], 6),
                    "ci_hi_tie": round(a["ci_tie"]["hi"], 6), "ci_lo_det": round(a["ci"]["lo"], 6), "ci_hi_det": round(a["ci"]["hi"], 6),
                    "p_router_better_det": round(a["ci"]["p_router_better"], 4), "headroom_recovered_tie": round(a["recovered_tie"], 4),
                    "headroom_recovered_det": round(a["recovered"], 4)})
    return rows, csv


SUM_HEAD = ["lambda", "router regret tie-avg (det.)", "default picks (of scored)", "agree det", "agree tie-avg", "best fixed",
            "always SA/WF/MA", "router - best fixed tie-avg (det.)", "95% cat-bootstrap CI tie-avg (det.)",
            "headroom recovered tie-avg (det.)"]


def file_fingerprint():
    out = {}
    for d in EXISTING_RESULT_DIRS:
        fs = sorted(glob.glob(os.path.join(R.REPO, "results", d, "*.json")))
        h = hashlib.md5()
        for f in fs:
            h.update(hashlib.md5(open(f, "rb").read()).hexdigest().encode())
        out[d] = (len(fs), h.hexdigest())
    return out


def git_changed(paths):
    r = subprocess.run(["git", "diff", "--name-only", "HEAD", "--"] + paths, cwd=R.REPO, capture_output=True, text=True)
    return [x for x in r.stdout.split("\n") if x]


def main():
    os.makedirs(OUT, exist_ok=True)
    fp0 = file_fingerprint()
    L = []

    def emit(s=""):
        print(s)
        L.append(s)

    def table(h, rows):
        for l in R.render(h, rows):
            emit("  " + l)

    tasks = load("structured")
    tasks_f1 = load("f1")
    by = {t.tid: t for t in tasks}
    scored = scored_set(tasks)
    assert len(scored) == 11
    excluded = sorted(t.tid for t in tasks if not t.scored)
    assert len(excluded) == 13

    emit("=" * 128)
    emit("router_dist: BUCKET-BASED SOFT-COUNT BETA ROUTER  (pre-registered in results/router_dist/SPEC.md)")
    emit("=" * 128)
    emit("Model: posterior Beta(1+s, 1+n-s) per bucket x architecture; evidence unit = one category's mean quality (a fractional 'soft count').")
    emit("THIS IS A SOFT-COUNT BETA UPDATE, NOT AN EXACT BETA-BINOMIAL LIKELIHOOD (quality is fractional F1/structured, not Bernoulli outcomes).")
    emit("Probabilities/P(best) describe uncertainty relative to this benchmark's bucket evidence, not calibrated success probabilities for arbitrary CRM tasks.")
    emit(f"Scored (informative) tasks: {sorted(scored)}")
    emit(f"Regret-excluded all-zero quality ties (listed, not scored): {excluded}")
    emit("Metric: structured (headline). Cost: category-mean cost, bucket mean of those. Utility U = mu - lambda * c_hat.")
    emit("Headline = bucket definition 'width only', leave-one-category-out. Regrets show the TIE-AVERAGED figure first, deterministic in parentheses.")

    all_recs, csv_out, folds = {}, [], {}
    for label in DEFS:
        for prot in ("LOCO", "LOTO"):
            all_recs[(label, prot)], folds[(label, prot)] = cross_validate(tasks, label, prot)

    # ---------------------------------------------------------------- headline
    emit("\n" + "#" * 128)
    emit("1. HEADLINE: width-only buckets, LEAVE-ONE-CATEGORY-OUT")
    emit("#" * 128)
    rows, c = summary_rows(tasks, all_recs[(HEAD, "LOCO")], scored, "headline LOCO width")
    table(SUM_HEAD, rows); csv_out += c
    emit("\n   LEAVE-ONE-TASK-OUT, width-only (OPTIMISTIC: same-category twins stay in training)")
    rows, c = summary_rows(tasks, all_recs[(HEAD, "LOTO")], scored, "LOTO width (optimistic)")
    table(SUM_HEAD, rows); csv_out += c

    emit("\n   BASELINE regrets (mean over the 11 scored tasks) per lambda:")
    rows = []
    for lam in LAMBDAS:
        fx = R.fixed_regrets(tasks, lam, scored)
        f = {a: mean(fx[a].values()) for a in ARCHS}
        bf = min(ARCHS, key=lambda a: (f[a], PRIORITY[a]))
        rows.append([f"{lam:g}", R.f3(f["single_agent"]), R.f3(f["workflow"]), R.f3(f["multi_agent"]), f"{SHORT[bf]} {R.f3(f[bf])}"])
    table(["lambda", "always SA", "always WF", "always MA", "best fixed"], rows)

    # ---------------------------------------------------------------- per-task tables
    emit("\n   PER-TASK TABLE (headline, LOCO) at each lambda: posterior mean +/- SD and expected cost per architecture")
    per_task_csv = []
    for lam in LAMBDAS:
        emit(f"\n   lambda = {lam:g}")
        rows = []
        for t in tasks:
            r = all_recs[(HEAD, "LOCO")][lam][t.tid]
            cells = [f"{r['mu'][i]:.2f}±{r['sd'][i]:.2f} ${r['chat'][i]:.4f}" for i in range(3)]
            rows.append([t.tid, t.category[:26], t.width, t.depth, r["bucket"], r["n_cat"]] + cells +
                        [SHORT[r["pick"]], f"{r['P_best']:.2f}", f"[{r['gap_lo']:+.2f},{r['gap_hi']:+.2f}]" + (" NO CLEAR WINNER" if r["no_clear_winner"] else ""),
                         "DEFAULT:" + r["flag"] if r["is_default"] else r["flag"], SHORT[r["oracle"]] if t.scored else "-(tied)"])
            per_task_csv.append({"lambda": lam, "task_id": t.tid, "category": t.category, "width": t.width, "depth": t.depth,
                                 "bucket": r["bucket"], "n_training_categories": r["n_cat"],
                                 **{f"{SHORT[a]}_mu": round(float(r["mu"][i]), 4) for i, a in enumerate(ARCHS)},
                                 **{f"{SHORT[a]}_sd": round(float(r["sd"][i]), 4) for i, a in enumerate(ARCHS)},
                                 **{f"{SHORT[a]}_expected_cost": round(float(r["chat"][i]), 6) for i, a in enumerate(ARCHS)},
                                 "selected": r["pick"], "P_best": round(r["P_best"], 4), "gap_ci_lo": round(r["gap_lo"], 4),
                                 "gap_ci_hi": round(r["gap_hi"], 4), "no_clear_winner": r["no_clear_winner"], "evidence_flag": r["flag"],
                                 "is_default_pick": r["is_default"], "scored": t.scored,
                                 "oracle_architecture": r["oracle"] if t.scored else "tied (regret-excluded)"})
        table(["task", "category", "w", "d", "bucket", "n_cat", "SA mu±sd $", "WF mu±sd $", "MA mu±sd $", "pick", "P(best)",
               "95% CI gap U(sel)-U(best other)", "evidence", "oracle"], rows)
    R.write_csv(os.path.join(OUT, "per_task_headline_LOCO.csv"), per_task_csv)

    # ---------------------------------------------------------------- variants
    emit("\n" + "#" * 128)
    emit("2. BUCKET-DEFINITION VARIANTS (labelled; none is the headline).  LOCO first, LOTO (optimistic) below each.")
    emit("#" * 128)
    for label in DEFS:
        if label == HEAD:
            continue
        for prot in ("LOCO", "LOTO"):
            emit(f"\n   {label} -- {prot}{' (OPTIMISTIC)' if prot == 'LOTO' else ''}")
            rows, c = summary_rows(tasks, all_recs[(label, prot)], scored, f"{SIMPLE[label]} {prot}")
            table(SUM_HEAD, rows); csv_out += c

    # ---------------------------------------------------------------- sensitivities
    emit("\n" + "#" * 128)
    emit("3. SENSITIVITY ANALYSES (each labelled; headline definition width-only unless stated; LOCO, LOTO alongside)")
    emit("#" * 128)
    sens = [
        ("(a) exclude task 722 from scoring", {HEAD}, tasks, scored_set(tasks, [722]), None),
        ("(b) exclude custom tasks 900001/900002 from scoring", {HEAD}, tasks, scored_set(tasks, [900001, 900002]), None),
        ("(f) PRIVACY SENSITIVITY: exclude privacy tasks 1624/1654/1914/1936 from scoring (their rewards are confounded; see CLAUDE.md audit)",
         {HEAD}, tasks, scored_set(tasks, PRIV_TASKS), None),
    ]
    for title, labels, tk, sc, _ in sens:
        for label in labels:
            for prot in ("LOCO", "LOTO"):
                emit(f"\n   {title} [{SIMPLE[label]}, {prot}, {len(sc)} scored tasks]")
                rows, c = summary_rows(tk, all_recs[(label, prot)], sc, f"{title[:3]} {SIMPLE[label]} {prot}")
                table(SUM_HEAD, rows); csv_out += c
    by_f1 = {t.tid: t for t in tasks_f1}
    sc_f1 = scored_set(tasks_f1)
    emit(f"\n   scored set under F1: {sorted(sc_f1)} (structured: {sorted(scored)})")
    for label in (HEAD, "width+depth (exploratory)"):
        recs_f1 = {p: cross_validate(tasks_f1, label, p)[0] for p in ("LOCO", "LOTO")}
        recs_d = {p: cross_validate(tasks, label, p, unit="run")[0] for p in ("LOCO", "LOTO")}
        recs_e = {p: cross_validate(tasks, label, p, cost_mode="ratio")[0] for p in ("LOCO", "LOTO")}
        for p in ("LOCO", "LOTO"):
            emit(f"\n   (c) ORIGINAL F1 METRIC [{SIMPLE[label]}, {p}]")
            rows, c = summary_rows(tasks_f1, recs_f1[p], sc_f1, f"(c) F1 {SIMPLE[label]} {p}")
            table(SUM_HEAD, rows); csv_out += c
            emit(f"\n   (d) OVERCONFIDENT SENSITIVITY: TREATS REPEATED RUNS AS INDEPENDENT EVIDENCE [{SIMPLE[label]}, {p}]")
            rows, c = summary_rows(tasks, recs_d[p], scored, f"(d) runs {SIMPLE[label]} {p}")
            table(SUM_HEAD, rows); csv_out += c
            sd_cat = mean(r["sd"].mean() for r in all_recs[(label, p)][0.0].values() if r["n_cat"] > 0)
            sd_run = mean(r["sd"].mean() for r in recs_d[p][0.0].values() if r["n_cat"] > 0)
            emit(f"      mean posterior SD over populated buckets: category-level {sd_cat:.3f} vs run-level {sd_run:.3f} (run-level is narrower: overconfident)")
            emit(f"\n   (e) RATIO-ANCHORED COST (router.py estimator, k=2) [{SIMPLE[label]}, {p}]")
            rows, c = summary_rows(tasks, recs_e[p], scored, f"(e) ratio-cost {SIMPLE[label]} {p}")
            table(SUM_HEAD, rows); csv_out += c

    # ---------------------------------------------------------------- comparison with router.py
    emit("\n" + "#" * 128)
    emit("4. COMPARISON WITH THE EXISTING router.py (picks over all 24 tasks; scored-11 listed)")
    emit("#" * 128)
    cmp_csv = []
    pairs = [("width+privacy_category_flag (diagnostic; leaks category)", "true", "like-for-like: same (width, has_refusal_check) buckets"),
             ("width+privacy_LLM_flag (diagnostic)", "llm", "like-for-like: same (width, LLM-only refusal flag) buckets"),
             (HEAD, "true", "NOT like-for-like: width-only has no counterpart in router.py (shown against the true-flag router)")]
    for label, setting, note in pairs:
        for ename, est in (("zero-target", R.EmpiricalShrinkageEstimator), ("pooled k=2", R.PooledPriorEstimator)):
            for prot in ("LOCO", "LOTO"):
                old = R.make_predictions(tasks, prot, setting, 2, est)
                emit(f"\n   {SIMPLE[label]} ({note}) vs existing {ename} ({setting} flag), {prot}")
                rows = []
                for lam in LAMBDAS:
                    ch_s, same_s, ch_o, same_o = [], [], [], []
                    for t in tasks:
                        o = R.task_result(t, old[t.tid], lam)["pick"]
                        n_ = all_recs[(label, prot)][lam][t.tid]["pick"]
                        (same_s if o == n_ else ch_s).append((t.tid, SHORT[o], SHORT[n_])) if t.scored else (same_o if o == n_ else ch_o).append(t.tid)
                        cmp_csv.append({"new_variant": SIMPLE[label], "existing": f"{ename}/{setting}", "protocol": prot, "lambda": lam,
                                        "task_id": t.tid, "scored": t.scored, "existing_pick": o, "new_pick": n_, "same": o == n_})
                    verdict = "IDENTICAL picks on all 24 tasks" if not ch_s and not ch_o else ""
                    rows.append([f"{lam:g}", f"{len(same_s)}/11 same", "; ".join(f"{a}: {b}->{c}" for a, b, c in ch_s) or "-", f"{len(ch_o)}/13 changed", verdict])
                table(["lambda", "scored same", "scored changed (task: existing->new)", "unscored", ""], rows)
    R.write_csv(os.path.join(OUT, "comparison_with_router_py.csv"), cmp_csv)

    # ---------------------------------------------------------------- sparsity & surprises
    emit("\n" + "#" * 128)
    emit("5. BUCKET SPARSITY UNDER LEAVE-ONE-CATEGORY-OUT (training categories available for each scored task)")
    emit("#" * 128)
    rows = []
    for label in DEFS:
        recs = all_recs[(label, "LOCO")][0.0]
        ns = {t.tid: recs[t.tid]["n_cat"] for t in tasks if t.scored}
        rows.append([SIMPLE[label], sum(1 for v in ns.values() if v == 0), sum(1 for v in ns.values() if v == 1),
                     sum(1 for v in ns.values() if v >= 2), ", ".join(f"{k}:{v}" for k, v in sorted(ns.items()))])
    table(["definition", "empty (of 11)", "n=1", "n>=2", "training categories per scored task"], rows)
    emit("   (the n=1 case has no cost SE: cost is treated as fixed in the simulation and flagged)")

    # ---------------------------------------------------------------- write CSV / report before final model
    R.write_csv(os.path.join(OUT, "summary_all.csv"), csv_out)

    # ---------------------------------------------------------------- final checks (part 1: before final fit)
    emit("\n" + "#" * 128)
    emit("6. FINAL CHECKS")
    emit("#" * 128)
    emit(f"   cross-validation folds fitted: " + ", ".join(f"{SIMPLE[k[0]]}/{k[1]}={v}" for k, v in folds.items()))
    emit("   held-out guard: every fold asserted no task of the held-out category (LOCO) / the held-out task (LOTO) was in training (assert passed).")
    changed = git_changed(PROTECTED)
    emit(f"   protected files/dirs modified vs HEAD (router.py, oracle.py, structured_score.py, normalize_score.py, decomposition/): {changed or 'none'}")
    fp1 = file_fingerprint()
    emit(f"   existing result dirs unchanged: {fp0 == fp1}  {fp1}")
    assert fp0 == fp1 and not changed

    # ---------------------------------------------------------------- final model (fitted AFTER evaluation)
    final = {"description": "FINAL bucket model fitted on all 24 tasks AFTER cross-validation was complete. "
                            "SOFT-COUNT Beta update (not an exact Beta-binomial likelihood). Evidence unit: task category.",
             "metric": "structured", "prior": "Beta(1,1)", "n_tasks_fitted": len(tasks), "n_categories": len({t.category for t in tasks}),
             "empty_bucket_cost_fallback_c_bar": None, "headline_definition": SIMPLE[HEAD], "bucket_model": None, "variants": {}}
    for label in DEFS:
        m = BetaBucketModel(DEFS[label]).fit(tasks)
        assert sum(len({t.category for t in tasks if DEFS[label](t) == k}) for k in m.buckets) >= len({t.category for t in tasks})
        entries = []
        for key in sorted(m.buckets):
            b = m.buckets[key]
            for i, a in enumerate(ARCHS):
                entries.append({"bucket_definition": SIMPLE[label], "bucket": key_str(label, key), "architecture": a, "n": b.n,
                                "soft_success_total_s": float(b.s[i]), "alpha": float(b.alpha[i]), "beta": float(b.beta[i]),
                                "posterior_mean": float(b.mu[i]), "posterior_sd": float(b.sd[i]),
                                "expected_cost": float(b.chat[i]),
                                "cost_se": None if b.se is None else float(b.se[i]),
                                "cost_se_available": b.se is not None})
        if label == HEAD:
            final["bucket_model"] = entries
            final["empty_bucket_cost_fallback_c_bar"] = float(m.c_bar)
        else:
            final["variants"][SIMPLE[label]] = entries
    path = os.path.join(OUT, "bucket_model.json")
    emit("\n   final model fitted on all 24 tasks:")
    rows = [[e["bucket"], e["architecture"], e["n"], f"{e['soft_success_total_s']:.3f}", f"{e['alpha']:.3f}", f"{e['beta']:.3f}",
             f"{e['posterior_mean']:.3f}", f"{e['posterior_sd']:.3f}", f"${e['expected_cost']:.5f}", e["cost_se"] if e["cost_se"] is None else f"{e['cost_se']:.5f}"]
            for e in final["bucket_model"]]
    table(["bucket", "architecture", "n", "s", "alpha", "beta", "mean", "sd", "exp. cost", "cost SE"], rows)
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(path, "w") as f:
        json.dump(final, f, indent=2)
    emit(f"\n   bucket_model.json written last (after all evaluation outputs); fitted on {final['n_tasks_fitted']} tasks / {final['n_categories']} categories.")
    with open(os.path.join(OUT, "report.txt"), "a") as f:
        f.write(f"\nbucket_model.json written last (after all evaluation outputs); fitted on {final['n_tasks_fitted']} tasks.\n")
    emit("   wrote: " + ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
