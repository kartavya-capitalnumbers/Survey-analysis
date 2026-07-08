"""
demographics.py — canned demographic breakdowns over the household survey dataset.

Per the spec: "Demographic breakdowns — by gender, age group, vulnerability
category, geographic zone." This module does not add new computation — it
orchestrates stats.py's frequency/crosstab/numeric_summary/compare_subsets
primitives into a fixed, curated set of breakdowns against those four
dimensions, so a practitioner gets the standard demographic cuts in one click
instead of building each one by hand in Quick Statistics.

Scoped to the exact column names household_extraction.analysis_row() produces
(the only source of survey_df in this app), so no column-existence guessing is
needed — every dimension referenced here is guaranteed present.
"""

from __future__ import annotations

from typing import List

import pandas as pd

import stats

# (dimension column, human label) — the four named breakdown axes.
DIMENSIONS = [
    ("head_gender", "Gender of household head"),
    ("head_age_group", "Age group of household head"),
    ("vulnerable_household", "Vulnerability category"),
    ("village", "Geographic zone (village)"),
]


def run_breakdowns(df: pd.DataFrame) -> List[stats.StatResult]:
    """Return the standard demographic-breakdown battery as a list of StatResults.

    Order: one frequency table per dimension, then a handful of illustrative
    cross-tabs and grouped numeric summaries linking those dimensions to the
    dataset's key outcome measures (income, food security).
    """
    results: List[stats.StatResult] = []

    for col, _label in DIMENSIONS:
        results.append(stats.frequency(df, col))

    def _try(fn, *args, **kwargs):
        try:
            results.append(fn(*args, **kwargs))
        except stats.StatsError:
            pass  # e.g. a dimension has too few valid values in a small dataset

    _try(stats.crosstab, df, "village", "vulnerable_household", percent="row")
    _try(stats.crosstab, df, "head_gender", "food_security_category", percent="row")
    _try(stats.numeric_summary, df, "monthly_income_usd", group_by="village")
    _try(stats.numeric_summary, df, "monthly_income_usd", group_by="head_gender")
    _try(stats.numeric_summary, df, "household_size", group_by="village")
    _try(stats.numeric_summary, df, "monthly_income_usd", group_by="vulnerable_household")

    return results
