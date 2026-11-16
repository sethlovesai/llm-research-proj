"""
structured_score.py -- deterministic structured scorer for the two custom tasks
(900001 churn_risk_assessment, 900002 dual_signal_account_flagging).

WHY: the stock fuzzy_match score (token-overlap F1 against one reference sentence)
rewards matching IDs, dates and boilerplate words, not correctness, on these tasks
(on 900002 single-agent stated "1" and "1" unresolved cases against the true 2 and 3
and still scored highest). This module scores the facts the ground truth states.

CONSTRAINTS: rules are derived from the ground-truth `answer` fields in
custom_tasks.py (verified against the live sandbox) and from hand-written examples
only. Deterministic regex parsing, no LLM judge. The same code scores every
architecture and every repeat. It never modifies result files: scores go to a
separate CSV via score_stored_results(), which must NOT be run until the rules
have been approved.

The text scored is the architecture's final answer (agent_info.end_reason.content),
the same field for single-agent, workflow and multi-agent.

=============================== 900002 RULE ===============================
Ground truth = 2 (account, unresolved-case count) pairs, N = 2:
    001Ws00003sIYPIIA4 (David Gonzalez) -> 2
    001Ws00003sIkv0IAC (Evelyn Hodges)  -> 3

  C = number of ground-truth accounts for which the answer states the correct count
  E = number of DISTINCT account IDs in the answer that are not ground-truth accounts

  score = max(0, (C - E) / N)          with N = 2

  * A wrong count on a ground-truth account earns no credit but carries no extra
    penalty. Only listing an account that is not in the ground truth is penalised
    (1/N per distinct extra ID). Extra accounts can be detected only when given as an
    account ID (001... , 15 or 18 chars); an unknown account NAME cannot be detected.
  * Order date/status is NOT scored (per the decision to score count pairs only).
  * Account identification: the 18-char ID, its 15-char prefix, or the full account
    name (case-insensitive). Adjacent mentions of the same account (e.g. "<id>
    (<name>)", gap <= 30 chars) are one mention.
  * Count reading: for each mention, take the text from its end to the start of the
    next account mention (or the end of the answer), blank out IDs/dates/years, then
    use the FIRST of these rules that matches anything:
        1. "<N> unresolved|open|outstanding|pending|active [support] case(s)"
        2. "unresolved|open [support] [case(s)] count|total|number [is|:|=] <N>"
           and "unresolved|open [support] case(s) [is|:|=] <N>"
        3. "case count|number of [unresolved] cases|case(s) [is|:|=] <N>"
        4. fallback: the standalone integers left in the segment
    <N> is digits or a number word (zero..twelve; not used by rule 4). If the winning
    rule yields more than one distinct value, the account is AMBIGUOUS: it gets no
    credit and the answer is flagged (never guessed). Two mentions of one account with
    different counts are also AMBIGUOUS.

=============================== 900001 RULE ===============================
The reference answer contains: a risk level (Elevated), a retention recommendation
(outreach recommended), two named unresolved High-priority cases (Engine
Malfunction, Warranty Repair Delay) and the last-order month (September 2023). It
contains NO case counts or order counts (it says "multiple"), so none are scored.
The task asks for a risk assessment and a retention recommendation; the named cases
and order recency are the evidence the reference gives for them. Five equally
weighted claims, N = 5:

  A  risk level: the answer asserts high/elevated churn risk
  B  retention action recommended: the answer recommends retention action/outreach
  C  names the unresolved High-priority case "Engine Malfunction"
  D  names the unresolved High-priority case "Warranty Repair Delay"
  E  states the last order was in September 2023 (any day/format: 2023-09-25,
     September 25 2023, Sept 2023, 09/2023 ...) in a sentence about an order/purchase

  C_ = number of claims A..E that are correct
  W  = number of claims that are CONTRADICTED; only A and B can be contradicted:
         A is contradicted if the answer asserts low/minimal/no churn risk (or
           negates high risk, e.g. "not at high risk");
         B is contradicted if the answer says retention action is not needed /
           not recommended / unnecessary.

  score = max(0, (C_ - W) / 5)

  * "Moderate/medium" risk is neither correct nor contradicted for A.
  * If an answer both asserts and negates the same claim (A or B) the claim is
    AMBIGUOUS: no credit, no penalty, and the answer is flagged.
  * Claims C, D, E have no contradiction test: a missing or wrong mention simply earns
    nothing. A wrong order date is not penalised.
  * Negation: a positive phrase counts as negated if not/no/never/hardly/unlikely
    occurs within the 3 words immediately before it.
  Sub-scores (decision claims A,B vs evidence claims C,D,E) are also returned for
  diagnostics; the headline score is the single five-claim figure above.

Usage:
    python3 structured_score.py --validate        # run the hand-written validation set
    python3 structured_score.py --score-stored    # ONLY after the rules are approved
"""
import argparse
import re

# --------------------------------------------------------------------------- shared
_NUMWORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_NUM = r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
          r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")


def _to_int(tok):
    return int(tok) if tok.isdigit() else _NUMWORDS[tok.lower()]


# ----------------------------------------------------------------------------- 900002
GT_900002 = {
    "001Ws00003sIYPIIA4": {"count": 2, "name": "David Gonzalez"},
    "001Ws00003sIkv0IAC": {"count": 3, "name": "Evelyn Hodges"},
}
_ID_RE = re.compile(r"(?<![A-Za-z0-9])001[A-Za-z0-9]{12}(?:[A-Za-z0-9]{3})?(?![A-Za-z0-9])")

_COUNT_RULES = [
    re.compile(rf"\b{_NUM}\s+(?:unresolved|open|outstanding|pending|active)\s+(?:support\s+)?cases?\b", re.I),
    re.compile(rf"\b(?:unresolved|open|outstanding)\s+(?:support\s+)?(?:cases?\s+)?(?:count|total|number)"
               rf"\s*(?:is|are|of|=|:|-)?\s*{_NUM}\b", re.I),
    re.compile(rf"\b(?:unresolved|open)\s+(?:support\s+)?cases?\s*(?:is|are|=|:|-)\s*{_NUM}\b", re.I),
    re.compile(rf"\b(?:case\s+count|number\s+of\s+(?:unresolved\s+|open\s+)?cases|cases?)"
               rf"\s*(?:is|are|=|:|-)\s*{_NUM}\b", re.I),
]
_STANDALONE_INT = re.compile(r"(?<![\w.\-/$])(\d+)(?![\w\-/%])(?!\.\d)")


def _mentions_900002(text):
    mentions = []
    for m in _ID_RE.finditer(text):
        tok = m.group(0)
        key = None
        for gid in GT_900002:
            if tok == gid or (len(tok) == 15 and gid.startswith(tok)):
                key = gid
                break
        mentions.append((m.start(), m.end(), key if key else "extra:" + tok[:15]))
    for gid, info in GT_900002.items():
        name_re = r"(?<![A-Za-z])" + r"\s+".join(re.escape(w) for w in info["name"].split()) + r"(?![A-Za-z])"
        for m in re.finditer(name_re, text, flags=re.I):
            mentions.append((m.start(), m.end(), gid))
    mentions.sort()
    merged = []
    for s, e, k in mentions:
        if merged and merged[-1][2] == k and s - merged[-1][1] <= 30:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), k)
        else:
            merged.append((s, e, k))
    return merged


def _blank_ids_dates_years(seg):
    t = _ID_RE.sub(" ", seg)
    t = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", " ", t)
    t = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", " ", t)
    t = re.sub(rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", " ", t, flags=re.I)
    t = re.sub(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?(?:,?\s+\d{{4}})?\b", " ", t, flags=re.I)
    t = re.sub(rf"\b{_MONTH}\.?\s+\d{{4}}\b", " ", t, flags=re.I)
    t = re.sub(r"\b(?:19|20)\d{2}\b", " ", t)
    return t


# Provenance labels, aligned with the rule numbering in the module docstring.
_RULE_LABELS = ["1:N_unresolved_cases", "2a:unresolved_count_label", "2b:unresolved_cases_label",
                "3:case_count_label"]
FALLBACK_LABEL = "4:fallback_sole_integer"


def _extract_count(seg):
    """Returns (value or None, ambiguous: bool, rule label that fired). The label is
    provenance only; it does not affect the value or the score."""
    t = _blank_ids_dates_years(seg)
    for label, rule in zip(_RULE_LABELS, _COUNT_RULES):
        vals = {_to_int(m.group(1)) for m in rule.finditer(t)}
        if vals:
            return ((next(iter(vals)), False, label) if len(vals) == 1 else (None, True, label))
    vals = {int(x) for x in _STANDALONE_INT.findall(t)}
    if len(vals) == 1:
        return next(iter(vals)), False, FALLBACK_LABEL
    if len(vals) > 1:
        return None, True, "4:fallback_multiple_integers"
    return None, False, "none"


def score_900002(text):
    text = text or ""
    blocks = _mentions_900002(text)
    per = {gid: {"status": "missing", "parsed": None, "rules": [], "fallback_used": False} for gid in GT_900002}
    seen = {gid: [] for gid in GT_900002}
    for i, (s, e, key) in enumerate(blocks):
        if key not in GT_900002:
            continue
        seg_end = blocks[i + 1][0] if i + 1 < len(blocks) else len(text)
        seen[key].append(_extract_count(text[e:seg_end]))
    flags = []
    C = 0
    for gid, results in seen.items():
        if not results:
            continue
        vals = {v for v, amb, _ in results if v is not None}
        any_amb = any(amb for _, amb, _ in results)
        rules = [rl for _, _, rl in results]
        prov = {"rules": rules, "fallback_used": any(rl == FALLBACK_LABEL for rl in rules)}
        if any_amb or len(vals) > 1:
            per[gid] = {"status": "ambiguous", "parsed": sorted(vals) or None, **prov}
            flags.append(f"{gid}: ambiguous count")
        elif len(vals) == 1:
            v = next(iter(vals))
            ok = v == GT_900002[gid]["count"]
            per[gid] = {"status": "correct" if ok else "wrong", "parsed": v, **prov}
            C += int(ok)
        else:
            per[gid] = {"status": "no_count", "parsed": None, **prov}
    extras = {k for _, _, k in blocks if k.startswith("extra:")}
    E = len(extras)
    N = len(GT_900002)
    return {
        "score": max(0.0, (C - E) / N), "C": C, "E": E, "N": N,
        "per_account": per, "extra_accounts": sorted(extras),
        "ambiguous": bool(flags), "flags": flags,
    }


# ----------------------------------------------------------------------------- 900001
_SIG = r"(?:high|elevated|significant|severe|serious|substantial|considerable|heightened)"
_LOWW = r"(?:low|minimal|minor|negligible|slight|small)"
_QUAL = r"(?:(?:overall|customer|account|attrition|churn)\s+)"
_RISK_COP = (r"\brisk\b(?:\s+of\s+churn)?(?:\s+level)?\s*(?:is|are|:|=|-)\s*"
             r"(?:currently\s+|considered\s+|assessed\s+as\s+|rated\s+(?:as\s+)?)?")

HIGH_PATTERNS = [
    re.compile(rf"\b{_SIG}\s+{_QUAL}{{0,2}}risk\b"),
    re.compile(rf"\b{_SIG}\s+(?:likelihood|probability|chance)\s+of\s+(?:churn|attrition|leaving|losing)\b"),
    re.compile(rf"\bat\s+(?:a\s+|an\s+)?{_SIG}\s+(?:\w+\s+)?risk\b"),
    re.compile(_RISK_COP + rf"{_SIG}\b"),
    re.compile(r"\b(?:very\s+|highly\s+)?likely\s+to\s+(?:churn|leave|cancel|defect)\b"),
]
LOW_PATTERNS = [
    re.compile(rf"\b{_LOWW}\s+{_QUAL}{{0,2}}risk\b"),
    re.compile(r"\bno\s+(?:(?:significant|real|apparent|immediate|churn)\s+){0,2}risk\b"),
    re.compile(r"\bnot\s+(?:currently\s+)?at\s+(?:any\s+)?risk\b"),
    re.compile(_RISK_COP + r"(?:low|minimal|minor|negligible)\b"),
    re.compile(r"\bunlikely\s+to\s+(?:churn|leave|cancel|defect)\b"),
]
_ADJ = r"(?:needed|recommended|required|warranted|advised|necessary|advisable|suggested|urgent|critical)"
RETAIN_POS = [
    re.compile(r"\b(?:retention|retain|outreach|intervention|re-?engagement|win-?back)\b"
               r"(?:\s+(?:action|actions|outreach|strategy|strategies|effort|efforts|plan|measures|steps|"
               r"program|campaign))?\s+(?:is|are|should\s+be|would\s+be|be|seems|appears)?\s*" + _ADJ + r"\b"),
    re.compile(r"\brecommend(?:s|ed|ing)?\b[^.;]{0,60}?\b(?:retention|retain|outreach|intervention|"
               r"re-?engag\w*|win-?back|proactive)\b"),
    re.compile(r"\b(?:should|must|need\s+to|needs\s+to|ought\s+to)\s+(?:\w+\s+){0,4}?"
               r"(?:retention|retain|outreach|reach\s+out|intervene|re-?engage)\b"),
]
RETAIN_NEG = [
    re.compile(r"\b(?:no|not)\b\s+(?:\w+\s+){0,3}?(?:retention|outreach|intervention|action)\b"
               r"(?:\s+\w+){0,4}?\s*" + _ADJ + r"\b"),
    re.compile(r"\b(?:retention|outreach|intervention|action)\b(?:\s+\w+){0,3}?\s+(?:(?:is|are|be)\s+)?not\s+"
               r"(?:\w+\s+)?" + _ADJ + r"\b"),
    re.compile(r"\bunnecessary\b|\bno\s+need\s+(?:for|to)\b|\bnot\s+recommended\b|\b(?:do|does)\s+not\s+recommend\b"),
]
_NEG_BEFORE = re.compile(r"\b(?:not|no|never|hardly|unlikely)\b(?:\s+\w+){0,3}\s*$")

_ORDER_WORD = re.compile(r"\b(?:orders?|ordered|purchases?|purchased)\b")
_SEPT_2023 = [
    re.compile(r"\b2023[-/]0?9(?:[-/]\d{1,2})?\b"),
    re.compile(r"\b0?9[-/](?:\d{1,2}[-/])?2023\b"),
    re.compile(r"\bsep(?:t(?:ember)?)?\.?\s+(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?2023\b"),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+sep(?:t(?:ember)?)?\.?,?\s+2023\b"),
]


def _norm(t):
    t = (t or "").lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        t = t.replace(a, b)
    t = re.sub(r"\b(is|are|was|were|do|does|did|has|have|had|should|would|could|must|need)n't\b", r"\1 not", t)
    t = t.replace("cannot", "can not").replace("can't", "can not").replace("won't", "will not")
    return re.sub(r"[ \t]+", " ", t)


def _sentences(t):
    return [s for s in re.split(r"(?<=[.!?;])\s+|\n+", t) if s.strip()]


def _risk_status(sents):
    pos = neg = amb = False
    for s in sents:
        for pat in HIGH_PATTERNS:
            for m in pat.finditer(s):
                if _NEG_BEFORE.search(s[:m.start()]):
                    neg = True            # e.g. "not at high risk"
                else:
                    pos = True
        for pat in LOW_PATTERNS:
            for m in pat.finditer(s):
                if re.search(r"\b(?:not|never|hardly)\b(?:\s+\w+){0,2}\s*$", s[:m.start()]):
                    amb = True            # e.g. "not a low risk": do not guess
                else:
                    neg = True
    if amb or (pos and neg):
        return "ambiguous"
    return "correct" if pos else ("contradicted" if neg else "unstated")


def _retain_status(sents):
    pos = neg = False
    for s in sents:
        for pat in RETAIN_NEG:
            if pat.search(s):
                neg = True
        for pat in RETAIN_POS:
            for m in pat.finditer(s):
                if _NEG_BEFORE.search(s[:m.start()]):
                    neg = True
                else:
                    pos = True
    if pos and neg:
        return "ambiguous"
    return "correct" if pos else ("contradicted" if neg else "unstated")


def score_900001(text):
    t = _norm(text)
    sents = _sentences(t)
    risk = _risk_status(sents)
    retain = _retain_status(sents)
    case_engine = bool(re.search(r"engine[\s_\-]+malfunction", t))
    case_warranty = bool(re.search(r"warranty[\s_\-]+repair[\s_\-]+delay", t))
    last_order = any(_ORDER_WORD.search(s) and any(p.search(s) for p in _SEPT_2023) for s in sents)
    claims = {
        "A_risk_high": risk, "B_retention_recommended": retain,
        "C_engine_malfunction": case_engine, "D_warranty_repair_delay": case_warranty,
        "E_last_order_sept_2023": last_order,
    }
    c_ok = int(risk == "correct") + int(retain == "correct") + int(case_engine) + int(case_warranty) + int(last_order)
    w = int(risk == "contradicted") + int(retain == "contradicted")
    N = 5
    flags = [f"{k}: ambiguous" for k, v in claims.items() if v == "ambiguous"]
    return {
        "score": max(0.0, (c_ok - w) / N), "C": c_ok, "W": w, "N": N, "claims": claims,
        "decision_correct": int(risk == "correct") + int(retain == "correct"),
        "evidence_correct": int(case_engine) + int(case_warranty) + int(last_order),
        "ambiguous": bool(flags), "flags": flags,
    }


def structured_score(task_id, text):
    if task_id == 900001:
        return score_900001(text)
    if task_id == 900002:
        return score_900002(text)
    raise ValueError(f"No structured scorer for task_id={task_id}")


# ------------------------------------------------------------------------- validation
GT_ANSWERS = {
    900001: ("Elevated churn risk. The account has multiple long-unresolved cases, including a High-priority "
             "'Engine Malfunction' case open since July 2021 and a High-priority 'Warranty Repair Delay' case "
             "open since October 2023, and no new order since September 2023. Retention outreach is recommended."),
    900002: ("Two accounts match both criteria. Account 001Ws00003sIYPIIA4 (David Gonzalez) has 2 unresolved "
             "cases; most recent order 2024-05-01 (Activated). Account 001Ws00003sIkv0IAC (Evelyn Hodges) has 3 "
             "unresolved cases; most recent order 2024-05-01 (Activated)."),
}

# (task_id, label, text, expected_score, expected_ambiguous). Expected values were fixed by
# hand BEFORE running the scorer.
VALIDATION = [
    (900002, "ground-truth answer", GT_ANSWERS[900002], 1.0, False),
    (900002, "good: id: N unresolved cases (lines)",
     "001Ws00003sIYPIIA4: 2 unresolved cases, latest order 2024-05-01 (Activated)\n"
     "001Ws00003sIkv0IAC: 3 unresolved cases, latest order 2024-05-01 (Activated)", 1.0, False),
    (900002, "good: terse csv-style",
     "001Ws00003sIYPIIA4, 2, 2024-05-01, Activated; 001Ws00003sIkv0IAC, 3, 2024-05-01, Activated", 1.0, False),
    (900002, "good: markdown table",
     "| AccountId | Unresolved Cases | Latest Order | Status |\n|---|---|---|---|\n"
     "| 001Ws00003sIYPIIA4 | 2 | 2024-05-01 | Activated |\n| 001Ws00003sIkv0IAC | 3 | 2024-05-01 | Activated |",
     1.0, False),
    (900002, "good: labelled fields",
     "AccountId: 001Ws00003sIYPIIA4, Unresolved Case Count: 2, Most Recent Order Date: 2024-05-01, Status: Activated; "
     "AccountId: 001Ws00003sIkv0IAC, Unresolved Case Count: 3, Most Recent Order Date: 2024-05-01, Status: Activated",
     1.0, False),
    (900002, "good: names only", "David Gonzalez has 2 unresolved cases; Evelyn Hodges has 3 unresolved cases.", 1.0, False),
    (900002, "good: number words",
     "001Ws00003sIYPIIA4 has two unresolved cases and 001Ws00003sIkv0IAC has three unresolved cases.", 1.0, False),
    (900002, "good: 15-char id prefixes",
     "001Ws00003sIYPI: 2 unresolved cases; 001Ws00003sIkv0: 3 unresolved cases", 1.0, False),
    (900002, "partial: only first account (correct)",
     "Account 001Ws00003sIYPIIA4 has 2 unresolved cases; most recent order 2024-05-01 (Activated).", 0.5, False),
    (900002, "partial: one count right, one wrong",
     "001Ws00003sIYPIIA4: 2 unresolved cases. 001Ws00003sIkv0IAC: 1 unresolved case.", 0.5, False),
    (900002, "WRONG: '1 and 1' counts (right ids, dates, status)",
     "001Ws00003sIYPIIA4: 1, 2024-05-01, Activated; 001Ws00003sIkv0IAC: 1, 2024-05-01, Activated", 0.0, False),
    (900002, "WRONG: counts swapped",
     "001Ws00003sIYPIIA4 has 3 unresolved cases; 001Ws00003sIkv0IAC has 2 unresolved cases", 0.0, False),
    (900002, "penalty: both correct + 1 invented account",
     "001Ws00003sIYPIIA4: 2 unresolved cases; 001Ws00003sIkv0IAC: 3 unresolved cases; "
     "001Ws00003sABCDEFG: 4 unresolved cases", 0.5, False),
    (900002, "penalty: one correct + 1 invented account",
     "001Ws00003sIYPIIA4: 2 unresolved cases; 001Ws00003sABCDEFG: 4 unresolved cases", 0.0, False),
    (900002, "WRONG: no accounts", "No accounts match both criteria.", 0.0, False),
    (900002, "flagged: ambiguous count (2 and 3 for one account)",
     "001Ws00003sIYPIIA4 has 2 unresolved cases and 3 open cases", 0.0, True),
    (900001, "ground-truth answer", GT_ANSWERS[900001], 1.0, False),
    (900001, "good + evidence",
     "The account is at high risk of churn: it has unresolved High-priority cases (Engine Malfunction and Warranty "
     "Repair Delay), and its last order was on 2023-09-25. Retention action is recommended.", 1.0, False),
    (900001, "partial: decisions only", "High churn risk. Recommend retention outreach.", 0.4, False),
    (900001, "partial: decisions, other wording",
     "Recommend: proactive retention outreach is warranted given elevated risk of churn.", 0.4, False),
    (900001, "partial: 'churn risk: high'", "Churn risk: high. Retention action needed.", 0.4, False),
    (900001, "partial: recommendation only", "Retention action is needed.", 0.2, False),
    (900001, "partial: evidence only",
     "Open High-priority cases: Engine Malfunction and Warranty Repair Delay. The last order was placed on "
     "September 25, 2023.", 0.6, False),
    (900001, "WRONG: low risk, no action", "Low churn risk. No retention action is needed.", 0.0, False),
    (900001, "WRONG: high risk but no action (1 right, 1 contradicted)",
     "Elevated churn risk, but no retention action is required.", 0.0, False),
    (900001, "WRONG: negation trap", "The account is not at high risk of churn.", 0.0, False),
    # Added with parser v2 (hand-written): negated recommendation with no copula. A=correct (+1),
    # B=contradicted (W=1) -> (1-1)/5 = 0.0; under v1 B was misread as unstated -> 0.2.
    (900001, "WRONG: 'retention action not needed' (no copula)",
     "Elevated churn risk. Retention action not needed.", 0.0, False),
    (900001, "neither: moderate risk", "Moderate churn risk; keep monitoring the account.", 0.0, False),
    (900001, "flagged: contradictory risk statements", "High risk. Actually, low risk.", 0.0, True),
]


def validate():
    bad = 0
    print(f"{'task':>7}  {'expected':>8} {'got':>6}  {'amb(exp/got)':>12}  {'ok':>3}  label")
    for tid, label, text, exp, exp_amb in VALIDATION:
        r = structured_score(tid, text)
        ok = abs(r["score"] - exp) < 1e-9 and r["ambiguous"] == exp_amb
        bad += not ok
        print(f"{tid:>7}  {exp:8.2f} {r['score']:6.2f}  {str(exp_amb):>5}/{str(r['ambiguous']):<6}  "
              f"{'OK' if ok else 'XX':>3}  {label}")
        if not ok:
            print("          detail:", {k: v for k, v in r.items() if k not in ('score',)})
    print(f"\n{len(VALIDATION) - bad}/{len(VALIDATION)} validation cases as expected")
    return bad == 0


# --------------------------------------------------------- scoring stored results (later)
PARSER_VERSION = "v2"   # bump on every parser fix and log it in structured_score_changelog.md


def score_stored_results(out_path="structured_scores.csv"):
    """Scores the stored final answers of both custom tasks for all three architectures
    and writes a SEPARATE csv (result files are never modified). The csv sits at the
    repo root because results*/ is gitignored. Columns: the original F1 score
    (untouched), `structured_score_v1` (the score under the first parser version, carried
    over unchanged from any earlier csv so pre-fix scores are never lost),
    `structured_score` (current parser) and `parser_version`."""
    import csv
    import json
    import os
    import oracle
    from normalize_score import normalize_score

    prev = {}
    if os.path.exists(out_path):
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                prev[(int(row["task_id"]), row["architecture"], int(row["run_number"]))] = row

    rows = []
    for tid in (900001, 900002):
        for arch in oracle.RESULT_GLOBS:
            for run in oracle._load_runs(tid, arch):
                r = run["result"]
                if run["infra_error"]:
                    continue
                answer = str((r["agent_info"].get("end_reason") or {}).get("content", ""))
                sc = structured_score(tid, answer)
                detail = json.dumps({k: v for k, v in sc.items() if k not in ("score", "flags", "ambiguous")})
                key = (tid, arch, r["run_number"])
                old = prev.get(key)
                rows.append({
                    "task_id": tid, "architecture": arch, "run_number": r["run_number"],
                    "f1_score": round(normalize_score(r), 6),
                    "structured_score_v1": old["structured_score_v1"] if old else round(sc["score"], 6),
                    "structured_score": round(sc["score"], 6),
                    "parser_version": PARSER_VERSION,
                    "ambiguous": sc["ambiguous"], "flags": "; ".join(sc["flags"]),
                    "detail_v1": old["detail_v1"] if old else detail,
                    "detail": detail,
                    "final_answer": answer.replace("\n", " \\n "),
                })
    rows.sort(key=lambda x: (x["task_id"], x["architecture"], x["run_number"]))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--score-stored", action="store_true")
    args = ap.parse_args()
    if args.validate:
        raise SystemExit(0 if validate() else 1)
    if args.score_stored:
        score_stored_results()
        print(f"wrote structured_scores.csv (parser {PARSER_VERSION})")
