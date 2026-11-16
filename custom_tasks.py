"""
Author-designed custom tasks that don't exist in the CRMArena-Pro B2C dataset
(TASKS_B2C, loaded in crm_sandbox/data/assets.py from the Salesforce/CRMArenaPro
"b2c" split -- 2140 real tasks, idx range 0-2139).

Integration mechanism: this module extends TASKS_B2C IN PLACE (list.extend, never
reassignment) rather than modifying crm_sandbox/data/assets.py or any consumer.
Every existing consumer that already does
`from crm_sandbox.data.assets import TASKS_B2C` -- workflow.py's _load_task,
multi_agent.py (via workflow.py's _load_task), run_custom_tasks.py -- picks these
up automatically, because a `from X import NAME` binds a reference to the same
list object X.NAME; mutating that object in place (not rebinding the name) makes
the change visible through every existing reference regardless of import order,
as long as this module has been imported at least once before a task_id lookup
happens. No changes needed to decompose_task(), workflow.py, or multi_agent.py.

Usage: `import custom_tasks` (side effect only) before calling decompose_task()/
run_workflow()/run_multi_agent() on task_ids 900001+.

task_ids use a 900001+ numbering scheme -- safely out of range of the real
dataset's idx values (0-2139 as of this writing; re-verify against
`max(t["idx"] for t in TASKS_B2C)` if the underlying HF dataset is ever updated).

Both tasks use reward_metric="fuzzy_match", not exact_match -- confirmed by
reading crm_sandbox/env/env.py's Evaluator.evaluate(): exact_match only handles
either a single literal string match, or an LLM-extracted, sorted LIST of atomic
values (IDs/short labels) compared for set equality. Neither task's answer is a
flat list of atomic values (Task A is an open-ended recommendation; Task B is a
compound, multi-account, multi-field answer), so exact_match's parse_answers()
extraction (tuned for short IDs) would either mangle them or require a bespoke
extraction branch -- i.e. real special-casing. fuzzy_match's free-text F1/BLEU/
ROUGE comparison against a single reference string handles both natively, with
no pipeline changes, the same way sales_insight_mining/knowledge_qa already use
it for non-trivial answers.

Neither task_type is added to EXTERNAL_FACING_TASKS (crm_sandbox/data/assets.py)
-- both are internal account-analysis tasks, not customer-facing disclosure
scenarios, so they correctly default to agent_type="internal" in workflow.py/
run_custom_tasks.py as-is.

GROUND TRUTH STATUS: both `answer` values were computed from real queries against
the live B2C sandbox (see the queries in each task's inline comment) and manually
verified against the raw query output on 2026-09-24 -- confirmed, not drafts.
"""
from crm_sandbox.data.assets import TASKS_B2C

# Copied verbatim from a real TASKS_B2C entry's metadata["optional"] (e.g. task
# 722) -- this text is identical across every real task inspected, so it appears
# to be fixed org-wide domain context, not task-specific. Only consumed by
# ChatAgent when eval_mode="aided" (crm_sandbox/agents/chat_agent.py's reset());
# harmless to include for schema completeness even under the default eval mode,
# which ignores it.
_DOMAIN_DETAILS_METADATA = (
    "# Domain Details\n"
    "## Quarters of the Year\n"
    "- Q1: January 1 to March 31 (both inclusive).\n"
    "- Q2: April 1 to June 30 (both inclusive).\n"
    "- Q3: July 1 to September 30 (both inclusive).\n"
    "- Q4: October 1 to December 31 (both inclusive).\n"
    "## Seasons\n"
    "- Winter (2024): December 1, 2024 to February 28/29, 2025 (both inclusive), "
    "specifying the year to avoid ambiguity.\n"
    "- Spring: March 1 to May 31 (both inclusive).\n"
    "- Summer: June 1 to August 31 (both inclusive).\n"
    "- Autumn/Fall: September 1 to November 30 (both inclusive).\n"
    "## Time Periods\n"
    "- Past 2 quarters: This refers to any timeframe spanning two quarters back "
    "from a specified `end_date`. (i.e. begins on the date calculated by "
    "subtracting six months from the end_date and ends on the end_date) That "
    "translates to a six-month period retrospectively from the `end_date`.\n"
    "- Issue Significantly More Than Other Months: This means there is a month "
    "where the number of cases reported are larger than all other months."
)

CUSTOM_TASKS = [
    {
        "idx": 900001,
        "task": "churn_risk_assessment",
        "persona": "You are a customer success manager focused on proactive retention and account health.",
        "query": (
            "Assess this account's churn risk by reviewing their support case history "
            "and order/purchase trend, then recommend whether retention action is needed."
        ),
        "metadata": {
            "optional": _DOMAIN_DETAILS_METADATA,
            "required": (
                "\n        ## Churn risk assessment guide.\n"
                "        Review the account's Case history (open vs. closed, priority, recency) and\n"
                "        Order history (frequency, recency) to assess churn risk. An unresolved case\n"
                "        is one whose ClosedDate is not set -- do not rely on any particular Status\n"
                "        value, since the exact set of Status labels used for an open case is not fixed.\n"
                "        \n"
                "        - Account Id to be considered is: 001Ws00003sIYQrIAO\n"
                "        "
            ),
        },
        "reward_metric": "fuzzy_match",
        # VERIFIED via live sandbox queries (SELECT Subject, Status, Priority, CreatedDate,
        # ClosedDate FROM Case WHERE AccountId = '001Ws00003sIYQrIAO'; SELECT EffectiveDate,
        # Status FROM Order WHERE AccountId = '001Ws00003sIYQrIAO'), raw output manually
        # confirmed 2026-09-24. Account "Sophia Bailey": 7 cases / 3 unresolved
        # (ClosedDate=null), 3 orders (2020-03-20, 2021-07-01, 2023-09-25). Two long-open,
        # High-priority cases: "Engine Malfunction" (open since 2021-07-08) and "Warranty
        # Repair Delay" (open since 2023-10-05, no order since). The third unresolved case
        # ("Warranty Terms Confusion", open since 2020-03-26) is Low priority, not High.
        "answer": [
            "Elevated churn risk. The account has multiple long-unresolved cases, including "
            "a High-priority 'Engine Malfunction' case open since July 2021 and a "
            "High-priority 'Warranty Repair Delay' case open since October 2023, and no new "
            "order since September 2023. Retention outreach is recommended."
        ],
    },
    {
        "idx": 900002,
        "task": "dual_signal_account_flagging",
        "persona": "You are an operations analyst monitoring account risk signals across the customer base.",
        "query": (
            "Which accounts currently have both at least one unresolved support case and "
            "at least one order in the last 90 days? For each matching account, return the "
            "unresolved case count and most recent order date/status."
        ),
        "metadata": {
            "optional": _DOMAIN_DETAILS_METADATA,
            "required": (
                "\n        ## Dual-signal account flagging guide.\n"
                "        'Unresolved' means the Case's ClosedDate field is not set (null) -- do not\n"
                "        rely on any particular Status value, since the exact set of Status labels used\n"
                "        for an open case is not fixed.\n"
                "        For this task, treat 'the last 90 days' as the fixed window\n"
                "        2024-03-17 to 2024-06-15 (both inclusive) -- anchored to the period actually\n"
                "        populated in this org's Order data (max EffectiveDate = 2024-06-15), not\n"
                "        today's date, since a relative date filter returns zero rows against this\n"
                "        sandbox's historical test data.\n"
                "        "
            ),
        },
        "reward_metric": "fuzzy_match",
        # VERIFIED via live sandbox queries, raw output manually confirmed 2026-09-24:
        #   SELECT AccountId FROM Case WHERE ClosedDate = null   (132 accounts with an
        #                                                          unresolved case)
        #   SELECT AccountId, EffectiveDate, Status FROM Order
        #     WHERE EffectiveDate >= 2024-03-17 AND EffectiveDate <= 2024-06-15
        #                                                          (7 accounts with an
        #                                                           order in the window)
        # Intersection (computed in Python, not SOQL): exactly 2 accounts. Confirmed
        # correct against the raw per-query output for both branches.
        "answer": [
            "Two accounts match both criteria. Account 001Ws00003sIYPIIA4 (David Gonzalez) "
            "has 2 unresolved cases; most recent order 2024-05-01 (Activated). Account "
            "001Ws00003sIkv0IAC (Evelyn Hodges) has 3 unresolved cases; most recent order "
            "2024-05-01 (Activated)."
        ],
    },
]

TASKS_B2C.extend(CUSTOM_TASKS)
