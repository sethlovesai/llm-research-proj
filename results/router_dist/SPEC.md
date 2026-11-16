# router_dist: bucket-based Beta-posterior router (PRE-REGISTRATION)

Written and committed BEFORE `router_dist.py` exists or has been run. Nothing in this file is
changed after router_dist results are seen. A genuine implementation bug found later is recorded in
`results/router_dist/CHANGELOG.md` (original behaviour, corrected behaviour, reason); the
specification itself stays as written.

What I have and have not seen at the time of writing: I have seen the bucket MEMBERSHIP for every
bucket definition below (task IDs and categories), and I know the earlier router/oracle results
(`router.py`, `oracle.py`, their reports). I have NOT seen any output of this new router.

## Context and status of the design choices
- On Oct 9 the supervisor (Alice) confirmed that a simple bucket-based Bayesian model is appropriate:
  group a new task with structurally similar historical tasks, estimate a Beta posterior of each
  architecture's quality in that bucket, use the posterior mean and uncertainty in the recommendation.
  She approved the simple bucket-based approach ONLY.
- The choice of structural features is the student's own and was not discussed with her.
- This router sits next to `router.py` and does not replace or modify it. The formal report's
  methodology will be rewritten later to match this implementation.

## Question
Does a bucket-based Beta-posterior router, using only DAG structure read from the cached
decomposition, pick architectures with lower utility regret than the best fixed architecture, and how
uncertain are its recommendations?

## Model: SOFT-COUNT Beta update (not an exact Beta-binomial likelihood)
This is a soft-count update. Evidence values are category-mean quality scores in [0,1] (structured
or F1 quality, which can be fractional), treated as fractional success counts. It is NOT an exact
Beta-binomial likelihood, because the evidence is not a set of independent Bernoulli outcomes. The same
statement is repeated in every report that shows these posteriors.

- **Evidence unit: one task category.** For each training category and architecture: the mean
  quality over all tasks and repeats of that category (every task has the same number of repeats, so
  this equals the mean of per-task means; the code asserts it). Value in [0,1] = soft success count.
- For bucket `b` and architecture `k`: `n_b` = number of distinct training categories in the bucket;
  `s_bk` = sum over those categories of the category-mean quality of `k`.
- Prior Beta(1,1). Posterior Beta(alpha = 1 + s_bk, beta = 1 + n_b - s_bk).
- `mu_bk = (1 + s_bk) / (2 + n_b)`; `sigma_bk = sqrt(mu_bk (1 - mu_bk) / (n_b + 3))`.
- **Empty bucket (n_b = 0):** Beta(1,1) for every architecture (mu = 0.5, sigma = sqrt(0.25/3));
  the recommendation is flagged "no evidence".

## Cost model
- For each bucket and architecture: the category-mean cost (mean over the category's tasks and repeats
  of per-run dollar cost, as stored); `c_hat_bk` = mean of those category-mean costs over the bucket's
  training categories.
- Standard error of `c_hat_bk` = sample SD (ddof = 1) of the category-mean costs / sqrt(n_b), reported.
  If n_b = 1 the SE is unavailable: the observed mean cost is treated as FIXED in the simulation and the
  prediction is flagged "cost SE unavailable".
- **Empty bucket cost:** the project's existing global fallback from `router.py`
  (`EmpiricalShrinkageEstimator.c_bar`): the mean over training tasks of each task's mean cost across the
  three architectures, one scalar used for all three. So in an empty bucket the three architectures have
  equal mu and equal cost, an exact three-way tie.

## Decision rule
- `U_k(lambda) = mu_k - lambda * c_hat_k`; select the architecture with the highest `U_k`.
- Exact ties (difference < 1e-9, as in `router.py`) are broken deterministically by: 1. lower expected
  cost; 2. single_agent; 3. workflow; 4. multi_agent. The decision itself is deterministic.
- A **default pick** is one made in an empty bucket or from an exact tie in `U`.
- **Tie-averaged regret:** every default pick is replaced by the average regret over the tied
  architectures. In the evaluation, the headline leads with the tie-averaged figure and shows the
  deterministic figure beside it; the number of default picks among the 11 scored tasks is reported.

## Uncertainty calculation (reported; never changes the pick)
5000 Monte Carlo draws, fixed seed: `numpy.random.default_rng([20261006, task_id])` (the seed constant
used by `router.py`), independent streams per task. In each draw: quality of each architecture ~
its fitted Beta posterior (independently); cost ~ Normal(c_hat, SE) clipped at 0 (fixed if SE is
unavailable). Utility per draw `U = quality - lambda * cost`.
Per task report: selected architecture; posterior mean quality; posterior SD; expected cost;
`P(best)` = share of draws in which the selected architecture has the highest `U` (ties in a draw
broken by the deterministic order above); and a 95% credible interval (2.5th-97.5th percentile over
draws) of the gap `U(selected) - max(U of the other two)`. Flag **"no clear winner"** when that interval
includes zero. This flag is descriptive and does not change the selection. The probabilities describe
uncertainty relative to this benchmark's bucket evidence, not calibrated success probabilities for
arbitrary CRM tasks.

## Bucket definitions
Features are read from the cached decomposition (`decomposition/cache/<id>.json`); no decomposition is
regenerated. width = `features.parallelism_width` (maximum number of subtasks that can run
concurrently); depth = `features.critical_path_depth` (longest dependency chain); an undecomposed
(atomic) task (722) has width = 1, depth = 1. No subtask count or other feature is added.

**HEADLINE: width only** (3 buckets: w1 = 20 tasks / 15 categories; w2 = 900001, 900002 / 2 categories;
w4 = 36, 89 / 1 category).

Why width only, and why depth is exploratory: depth was considered as a second structural feature but is
treated as exploratory because it fragments the small dataset and is strongly associated with task
category in this benchmark. Pre-result sparsity audit of `(width, depth)`: 7 buckets, five containing a
single category, which under leave-one-category-out leaves 7 of the 11 informative tasks (722, 36, 89,
900001, 900002, 1914, 1936) with an empty bucket. Within the privacy categories, depth separates the
categories (1624/1654 have depth 3, 1914/1936 depth 5). Width-only also matches the supervisor's original
suggestion of using accuracy plus width.

**Labelled comparison / exploratory variants** (none is the headline):
- `width + depth` (7 buckets): exploratory, for the reasons above.
- `width + category-derived privacy flag`: the same `has_refusal_check` definition as `router.py`
  (single root subtask that is an Analyst). It leaks benchmark-category information (the refusal check
  is injected from the category), so it is a diagnostic comparison only.
- `width + LLM-only privacy flag`: the `router.py` definition (category-derived flag AND NOT
  `llm_omitted_refusal_check`). Diagnostic comparison only.

## Evaluation
Same conventions and loaders as `router.py` (`load_tasks`, `oracle.oracle_for_task`, quality from the
structured metric for the headline, per-task regret/oracle conventions, task-level bootstrap helper).
- **Headline validation: leave-one-category-out (LOCO).** For each held-out category, the model is fitted
  on ALL tasks except that category's; its tasks are then predicted. The code asserts that no
  held-out-category task is in the training set.
- **Secondary: leave-one-task-out (LOTO)**, labelled OPTIMISTIC because same-category twins stay in
  training. Evidence units are recomputed from the remaining tasks (the held-out task is removed before
  category means are formed).
- **Final prototype model:** fitted only AFTER the cross-validation outputs are written, on all 24 tasks,
  and saved separately as `bucket_model.json` (headline definition at top level; the variants under
  `variants`). Per bucket and architecture: bucket definition, architecture, `n`, soft success total
  `s`, Beta alpha and beta, posterior mean, posterior SD, expected cost, cost SE (or an
  "unavailable" flag).
- **Evaluation population:** the 11 strict-quality-win informative tasks (`oracle.py`, structured metric).
  The 13 all-zero quality-tied tasks are listed separately as regret-excluded and are not mixed into the
  headline regret.
- **lambda in {0, 10, 100, 300}.** Regret = oracle utility minus the chosen architecture's actual utility
  (actual task quality minus lambda times actual task mean cost), averaged over scored tasks.
- **Baselines:** always single-agent, always workflow, always multi-agent, and the best fixed
  architecture at each lambda (in-sample, lowest mean regret over the scored tasks).
- **Reported:** mean regret (tie-averaged first, deterministic beside it); number of default picks;
  oracle agreement (pick equals the oracle's architecture at that lambda; tie-averaged agreement
  alongside); router minus best-fixed regret; the 95% category-bootstrap interval for that difference
  (5000 resamples of the scored tasks' categories, same helper logic as `router.py`); and the share of
  the best-fixed-versus-oracle headroom recovered, `1 - router_regret / best_fixed_regret`.
- Everything is descriptive: no numerical success threshold is set.

## Sensitivity analyses (each labelled in the outputs; training is unchanged unless stated)
a. Exclude task 722 from the scored set.
b. Exclude custom tasks 900001 and 900002 from the scored set (training unchanged).
c. Original F1 metric instead of the structured metric (quality for training and scoring; the oracle and
   scored set are recomputed under F1).
d. **"Overconfident sensitivity: treats repeated runs as independent evidence."** Every run of every
   training task is one evidence unit (quality of that run as a soft count; cost per run; SE across
   runs). It is expected to give narrower posteriors than the category-level headline.
e. Ratio-anchored cost estimator from `router.py`'s `EmpiricalShrinkageEstimator` (k = 2:
   per-task cost ratios, shrunk toward 1, re-anchored at the global mean cost) in place of the observed
   bucket mean cost; quality posteriors and simulated cost SE unchanged.
f. **Privacy sensitivity analysis:** exclude privacy tasks 1624, 1654, 1914 and 1936 from the scored set
   (training unchanged). Reported separately; their stored privacy rewards are known to be confounded by
   the forced refusal-check trajectory and the privacy judge's behaviour (see CLAUDE.md, privacy audit).
   It does not replace the main benchmark.

Sensitivities a, b, f are run for the headline definition; c, d, e are run for the headline definition
and for `width + depth`. All use LOCO with LOTO shown alongside.

## Comparison with the existing router
For directly comparable bucket definitions only (`width + category-derived privacy flag` and
`width + LLM-only privacy flag`, both defined identically to `router.py`'s two settings), list the tasks
whose pick changes versus the existing zero-target router and versus the existing pooled-prior (k = 2)
router, and the tasks whose pick is identical; state plainly if a variant yields exactly the same picks.
The headline width-only model has no counterpart in `router.py`, so its comparison is shown but labelled
NOT like-for-like. `router.py` is not modified.

## Outputs (all under `results/router_dist/`)
`SPEC.md`; per-task table (task, category, width, depth, bucket key, number of training categories,
posterior mean +/- SD and expected cost per architecture, selected architecture, `P(best)`,
uncertainty flag, default flag, oracle architecture); summary tables (LOCO headline, LOTO optimistic,
lambda sensitivity, fixed baselines, regret deterministic and tie-averaged, default count, oracle
agreement, bootstrap interval, headroom recovered, sensitivities, comparison with `router.py`);
`bucket_model.json`; `report.txt`; `CHANGELOG.md` only if a genuine bug is fixed.

## Known limitations (stated in advance)
- Very few informative tasks (11) and only 3 repeats per task; most oracle margins are about one
  standard error.
- Buckets are sparse: w4 contains one category and w2 two, so held-out predictions for those tasks have
  little or no evidence.
- Width (and depth) are partly confounded with task category; every category sits at exactly one width.
- The soft-count update is not an exact binomial likelihood; the posterior SD is a convenient summary, not
  a calibrated interval.
- Privacy-task scores are confounded (forced refusal-check trajectory in workflow/multi-agent; judge
  behaviour), and the category-derived privacy flag leaks the benchmark category.
- Probabilities and `P(best)` describe uncertainty relative to this benchmark, not calibrated success
  probabilities for arbitrary CRM tasks.

## Procedure rules
- This SPEC is committed before `router_dist.py` is written or run, and not altered afterwards.
- Not modified: `router.py`, `oracle.py`, `structured_score.py`, `normalize_score.py`, the decomposition
  cache, and every existing result file. New output goes only to `results/router_dist/`.
