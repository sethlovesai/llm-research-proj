"""
task_id -> complexity tier ("low" / "medium" / "high") for the 24-task benchmark
set (22 real CRMArena-Pro B2C tasks + custom tasks 900001/900002).

Provenance: this mapping is author-supplied (the project owner's own
categorization of the task set), not derived from any computed property of the
tasks or from the CRMArena-Pro dataset. It was recorded here on 2026-10-01
because no tier mapping file previously existed in the repo, despite CLAUDE.md
referring to "task selection notes". Treat it as an input to the analysis, not a
result of it.

The two custom tasks (900001, 900002) are tier "high" by design -- both were
authored specifically as parallel (width-2) tasks.

Consumers: oracle.py (per-task tier column + by-tier win counts); the router
should import TASK_TIERS from here too rather than redefining it.
"""

TASK_TIERS = {
    # Low
    722: "low",      # knowledge_qa
    1942: "low",     # named_entity_disambiguation
    672: "low",      # invalid_config
    1857: "low",     # quote_approval
    # Medium
    122: "medium",   # monthly_trend_analysis
    125: "medium",   # monthly_trend_analysis
    792: "medium",   # sales_amount_understanding
    1030: "medium",  # conversion_rate_comprehension
    1089: "medium",  # best_region_identification
    2056: "medium",  # sales_cycle_understanding
    1463: "medium",  # policy_violation_identification
    1563: "medium",  # case_routing
    # High
    446: "high",     # sales_insight_mining
    421: "high",     # sales_insight_mining
    334: "high",     # top_issue_identification
    348: "high",     # top_issue_identification
    1936: "high",    # internal_operation_data
    1914: "high",    # internal_operation_data
    1624: "high",    # private_customer_information
    1654: "high",    # private_customer_information
    36: "high",      # lead_qualification
    89: "high",      # lead_qualification
    # Custom (high, by design -- both parallel)
    900001: "high",  # churn_risk_assessment
    900002: "high",  # dual_signal_account_flagging
}

TIER_ORDER = ["low", "medium", "high"]

assert set(TASK_TIERS.values()) <= set(TIER_ORDER), "unexpected tier label in TASK_TIERS"


def get_tier(task_id):
    """Tier for task_id, or "unknown" if the task isn't in the mapping."""
    return TASK_TIERS.get(task_id, "unknown")
