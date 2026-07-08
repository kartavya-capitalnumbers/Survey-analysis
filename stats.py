"""
stats.py — deterministic statistical analysis over a survey dataset.

This is the STATISTICAL counterpart to the narrative pipeline (schema/qa/report):
it operates on a tabular survey dataset — one row per respondent/form, one column
per survey field — the shape the future transcription layer will produce and that
a practitioner can already supply today as CSV/Excel.

Design rule (mirrors the grounding ethos of qa.py): the LLM NEVER computes a
number. It is used only to
  1. translate a plain-English analysis question into a structured operation
     spec (`plan_query`), and
  2. narrate an already-computed result table in plain English (`narrate_result`).
All arithmetic is done deterministically by pandas, and every result carries its
sample size and a methodology note so tables are audit-ready per the spec
("labelled clearly with sample size, methodology notes").

Supported operations (the Phase 1 statistical outputs):
  - frequency        counts + percentages for one field
  - crosstab         cross-tabulation of two fields (counts, optional row/col/overall %)
  - numeric_summary  mean / median / min / max / range / std, optionally grouped
All three accept row filters, which is what makes demographic breakdowns and
subset questions ("female-headed households in Zone B") expressible.

PROTOTYPE NOTE
--------------
The dataset comes from household_extraction.py (PDF survey forms transcribed via
pdfplumber/OCR + one Bedrock call per form, assembled into a DataFrame); this
module only ever sees the resulting rows, never a file.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import llm_client
import schema as schema_mod


class StatsError(RuntimeError):
    """Raised for dataset/plan problems so the UI can show them cleanly."""


# ---------------------------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------------------------

def profile_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Describe each column (kind, missing, sample values) for UI + planner."""
    out: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        else:
            nunique = s.nunique(dropna=True)
            kind = "categorical" if nunique <= max(20, len(s) // 10) else "text"
        sample = [str(v) for v in s.dropna().unique()[:5]]
        out.append(
            {
                "name": col,
                "kind": kind,
                "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True)),
                "sample_values": sample,
            }
        )
    return out


def _require_column(df: pd.DataFrame, name: Any) -> str:
    if not isinstance(name, str) or name not in df.columns:
        raise StatsError(
            f"Column {name!r} not found. Available: {', '.join(df.columns)}"
        )
    return name


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StatResult:
    title: str
    table: pd.DataFrame
    n: int                      # rows analysed (after filters, before dropping missing)
    methodology: str
    caveats: List[str] = field(default_factory=list)

    def table_text(self) -> str:
        """Plain-text rendering for LLM narration context."""
        return self.table.to_string(index=False)

    def to_csv(self) -> str:
        header = (
            f"# {self.title}\n# n = {self.n}\n# Methodology: {self.methodology}\n"
            + "".join(f"# Caveat: {c}\n" for c in self.caveats)
        )
        return header + self.table.to_csv(index=False)

    def to_excel_bytes(self) -> bytes:
        """Excel export with a metadata block (title, n, methodology, caveats)
        and a per-row confidence flag based on that row's sample size, per the
        spec's 'labelled clearly with sample size, methodology notes, and
        confidence flags' requirement.
        """
        table = self.table.copy()
        table["confidence_flag"] = _confidence_flags(table, self.n)
        buf = io.BytesIO()
        meta_lines = [self.title, f"n = {self.n}", f"Methodology: {self.methodology}"]
        meta_lines += [f"Caveat: {c}" for c in self.caveats]
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            pd.DataFrame({"": meta_lines}).to_excel(
                writer, sheet_name="Result", index=False, header=False, startrow=0
            )
            table.to_excel(writer, sheet_name="Result", index=False, startrow=len(meta_lines) + 1)
        return buf.getvalue()


def _confidence_flags(table: pd.DataFrame, overall_n: int) -> pd.Series:
    """Low/Moderate/OK per row, based on that row's own count if the table has
    one (a grouped breakdown's per-group size), else the table's overall n.
    """
    counts = table["count"] if "count" in table.columns else pd.Series([overall_n] * len(table))

    def _flag(c: Any) -> str:
        try:
            c = float(c)
        except (TypeError, ValueError):
            return ""
        if c < 5:
            return "Low (n<5) — indicative only"
        if c < 20:
            return "Moderate (n<20)"
        return "OK"

    return counts.apply(_flag)


# ---------------------------------------------------------------------------
# Row filters (shared by all operations)
# ---------------------------------------------------------------------------

_FILTER_OPS = ("==", "!=", ">", ">=", "<", "<=", "in", "contains")


def _apply_filters(
    df: pd.DataFrame, filters: Optional[List[Dict[str, Any]]]
) -> Tuple[pd.DataFrame, str]:
    """Apply row filters; return (subset, human-readable description)."""
    if not filters:
        return df, ""
    data = df
    notes: List[str] = []
    for f in filters:
        col = _require_column(data, f.get("field"))
        op = f.get("op")
        value = f.get("value")
        if op not in _FILTER_OPS:
            raise StatsError(f"Unsupported filter op {op!r} (use one of {_FILTER_OPS}).")
        s = data[col]
        if op in (">", ">=", "<", "<="):
            nums = pd.to_numeric(s, errors="coerce")
            try:
                v = float(value)
            except (TypeError, ValueError) as exc:
                raise StatsError(f"Filter value {value!r} is not numeric.") from exc
            mask = {
                ">": nums > v, ">=": nums >= v, "<": nums < v, "<=": nums <= v,
            }[op]
        elif op == "in":
            values = value if isinstance(value, list) else [value]
            wanted = {str(v).strip().lower() for v in values}
            mask = s.astype(str).str.strip().str.lower().isin(wanted)
        elif op == "contains":
            mask = s.astype(str).str.contains(str(value), case=False, na=False)
        else:  # == / != — numeric compare when both sides are numeric, else text
            nums = pd.to_numeric(s, errors="coerce")
            try:
                v_num: Optional[float] = float(value)
            except (TypeError, ValueError):
                v_num = None
            if v_num is not None and nums.notna().any():
                mask = nums == v_num
            else:
                mask = s.astype(str).str.strip().str.lower() == str(value).strip().lower()
            if op == "!=":
                mask = ~mask & s.notna()
        data = data[mask.fillna(False)] if mask.dtype == object else data[mask]
        notes.append(f"{col} {op} {value!r}")
    return data, "; ".join(notes)


def _filter_suffix(filter_note: str) -> str:
    return f" (filtered to: {filter_note})" if filter_note else ""


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def frequency(
    df: pd.DataFrame, field_name: str, *, filters: Optional[List[Dict[str, Any]]] = None
) -> StatResult:
    """Counts and percentages for one field. Percentages are of valid responses."""
    data, filter_note = _apply_filters(df, filters)
    col = _require_column(data, field_name)
    n = len(data)
    valid = data[col].dropna()
    n_valid = len(valid)
    if n_valid == 0:
        raise StatsError(f"No non-missing values in {col!r} after filtering.")
    counts = valid.astype(str).str.strip().value_counts()
    tab = pd.DataFrame(
        {
            col: counts.index,
            "count": counts.to_numpy(),
            "percent": (counts.to_numpy() / n_valid * 100).round(1),
        }
    )
    caveats = []
    n_missing = n - n_valid
    if n_missing:
        caveats.append(f"{n_missing} of {n} rows have no response for {col!r} and are excluded.")
    return StatResult(
        title=f"Frequency of {col}{_filter_suffix(filter_note)}",
        table=tab,
        n=n,
        methodology=(
            f"Counts of responses for '{col}'. Percentages are of the {n_valid} "
            f"valid (non-missing) responses."
        ),
        caveats=caveats,
    )


def crosstab(
    df: pd.DataFrame,
    row_field: str,
    col_field: str,
    *,
    percent: Optional[str] = None,   # None | "row" | "col" | "overall"
    filters: Optional[List[Dict[str, Any]]] = None,
) -> StatResult:
    """Cross-tabulate two fields. Optionally annotate cells with percentages."""
    data, filter_note = _apply_filters(df, filters)
    r = _require_column(data, row_field)
    c = _require_column(data, col_field)
    if percent not in (None, "row", "col", "overall"):
        raise StatsError("percent must be one of: row, col, overall.")
    n = len(data)
    sub = data.dropna(subset=[r, c])
    if sub.empty:
        raise StatsError(f"No rows have values for both {r!r} and {c!r}.")
    rows = sub[r].astype(str).str.strip()
    cols = sub[c].astype(str).str.strip()
    counts = pd.crosstab(rows, cols)

    if percent:
        if percent == "row":
            pct = counts.div(counts.sum(axis=1), axis=0) * 100
            denom = "its row total"
        elif percent == "col":
            pct = counts.div(counts.sum(axis=0), axis=1) * 100
            denom = "its column total"
        else:
            pct = counts / counts.to_numpy().sum() * 100
            denom = f"all {len(sub)} cross-tabulated responses"
        cells = counts.astype(str) + " (" + pct.round(1).astype(str) + "%)"
        method_pct = f" Each cell shows count (percent of {denom})."
    else:
        cells = counts.copy()
        cells["Total"] = counts.sum(axis=1)
        total_row = counts.sum(axis=0)
        total_row["Total"] = counts.to_numpy().sum()
        cells.loc["Total"] = total_row
        method_pct = " Margins are raw counts."

    tab = cells.reset_index().rename(columns={"index": r})
    tab.columns = [str(x) for x in tab.columns]
    caveats = []
    n_dropped = n - len(sub)
    if n_dropped:
        caveats.append(
            f"{n_dropped} of {n} rows are missing {r!r} and/or {c!r} and are excluded."
        )
    return StatResult(
        title=f"{r} × {c}{_filter_suffix(filter_note)}",
        table=tab,
        n=n,
        methodology=(
            f"Cross-tabulation of '{r}' (rows) by '{c}' (columns) over "
            f"{len(sub)} responses with both fields present.{method_pct}"
        ),
        caveats=caveats,
    )


def numeric_summary(
    df: pd.DataFrame,
    field_name: str,
    *,
    group_by: Optional[str] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
) -> StatResult:
    """Mean / median / min / max / range / std for a numeric field, optionally grouped."""
    data, filter_note = _apply_filters(df, filters)
    col = _require_column(data, field_name)
    n = len(data)
    nums = pd.to_numeric(data[col], errors="coerce")
    caveats: List[str] = []
    n_coerced = int(nums.isna().sum()) - int(data[col].isna().sum())
    if n_coerced > 0:
        caveats.append(f"{n_coerced} non-numeric values in {col!r} were treated as missing.")
    n_missing = int(nums.isna().sum())
    if n_missing:
        caveats.append(f"{n_missing} of {n} rows have no numeric value for {col!r} and are excluded.")
    if nums.notna().sum() == 0:
        raise StatsError(f"Column {col!r} has no numeric values to summarise.")

    def _agg(v: pd.Series) -> Dict[str, float]:
        return {
            "count": int(v.count()),
            "mean": round(float(v.mean()), 2),
            "median": round(float(v.median()), 2),
            "min": round(float(v.min()), 2),
            "max": round(float(v.max()), 2),
            "range": round(float(v.max() - v.min()), 2),
            "std": round(float(v.std()), 2) if v.count() > 1 else 0.0,
        }

    if group_by:
        g = _require_column(data, group_by)
        work = pd.DataFrame({g: data[g], "__v": nums}).dropna()
        if work.empty:
            raise StatsError(f"No rows have values for both {col!r} and {g!r}.")
        records = [
            {g: str(name), **_agg(grp["__v"])} for name, grp in work.groupby(g, sort=True)
        ]
        tab = pd.DataFrame(records)
        title = f"{col} by {g}{_filter_suffix(filter_note)}"
        methodology = (
            f"Summary statistics of '{col}' grouped by '{g}', computed over rows "
            f"with both fields present. std is the sample standard deviation."
        )
    else:
        tab = pd.DataFrame([{"field": col, **_agg(nums.dropna())}])
        title = f"Summary of {col}{_filter_suffix(filter_note)}"
        methodology = (
            f"Summary statistics of '{col}' over its valid numeric values. "
            f"std is the sample standard deviation."
        )
    return StatResult(title=title, table=tab, n=n, methodology=methodology, caveats=caveats)


def compare_subsets(
    df: pd.DataFrame,
    metric: str,
    by: str,
    *,
    filters: Optional[List[Dict[str, Any]]] = None,
) -> StatResult:
    """Rank subsets of `by` by mean `metric` — "trend identification across
    survey subsets" per the spec: without a repeated/longitudinal survey there
    is no time axis to trend against, so this surfaces the comparable signal
    that IS available — which subsets diverge on a metric, and by how much,
    ranked highest to lowest with each subset's gap from the top subset.
    """
    result = numeric_summary(df, metric, group_by=by, filters=filters)
    tab = result.table.sort_values("mean", ascending=False).reset_index(drop=True)
    tab.insert(0, "rank", range(1, len(tab) + 1))
    top_mean = tab["mean"].iloc[0]
    tab["pct_of_highest"] = (tab["mean"] / top_mean * 100).round(1) if top_mean else 0.0
    return StatResult(
        title=f"{metric} ranked by {by}",
        table=tab,
        n=result.n,
        methodology=(
            f"Subsets of '{by}' ranked by mean '{metric}', highest to lowest. "
            "'pct_of_highest' is each subset's mean as a percentage of the "
            "top-ranked subset's mean, showing the size of the gap between subsets."
        ),
        caveats=result.caveats,
    )


# ---------------------------------------------------------------------------
# Natural-language query planning (LLM translates; pandas computes)
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You translate a plain-English survey-analysis question into ONE statistical "
    "operation spec, as JSON. You never compute results yourself.\n\n"
    "Available operations:\n"
    '1. {"operation": "frequency", "field": "<column>", "filters": [...]}\n'
    "   — counts + percentages of one field's values.\n"
    '2. {"operation": "crosstab", "row_field": "<column>", "col_field": "<column>", '
    '"percent": null|"row"|"col"|"overall", "filters": [...]}\n'
    "   — cross-tabulation of two fields.\n"
    '3. {"operation": "numeric_summary", "field": "<column>", "group_by": '
    '"<column>"|null, "filters": [...]}\n'
    "   — mean/median/min/max/range/std of a numeric field, optionally per group.\n"
    '4. {"operation": "compare_subsets", "metric": "<numeric column>", "by": '
    '"<column>", "filters": [...]}\n'
    "   — ranks subsets of one column by mean of a numeric metric, highest to "
    "lowest, with each subset's percentage of the top subset's mean. Use this "
    "for trend/comparison questions across subsets/zones/groups (e.g. 'which "
    "village has the highest income', 'compare X across zones').\n"
    '5. {"operation": "narrative_summary"}\n'
    "   — a written, multi-paragraph socioeconomic-profile narrative (household "
    "composition, livelihoods/income, vulnerability, housing/land/services, "
    "community concerns), suitable for a RAP/ESIA/ESDD chapter. Use this "
    "whenever the question asks for an overall summary, profile, narrative, or "
    "write-up of the dataset as a whole rather than one specific figure — do "
    "NOT mark these 'unsupported'.\n\n"
    'Each optional filter is {"field": "<column>", "op": "=="|"!="|">"|">="|"<"|'
    '"<="|"in"|"contains", "value": <scalar or list>}.\n\n'
    "Rules:\n"
    "- Return ONLY the JSON object. No prose, no markdown fences.\n"
    "- Use EXACT column names from the provided column profile.\n"
    "- Percentage questions about one field -> frequency (it includes percentages). "
    "Use filters to restrict to the relevant subset instead of guessing.\n"
    "- Requests for an overall narrative/summary/profile -> narrative_summary, "
    "never 'unsupported'.\n"
    "- If the question cannot be answered with any of these operations on these "
    "columns (e.g. the data needed is entirely absent from the dataset), return "
    '{"operation": "unsupported", "reason": "<one sentence>"}.'
)


def build_plan_prompt(question: str, df: pd.DataFrame) -> Tuple[str, str]:
    """Return (system, user) for the query-planning call. Isolated for reuse."""
    profile = json.dumps(profile_columns(df), indent=2)
    user = (
        f"COLUMN PROFILE of the survey dataset ({len(df)} rows):\n{profile}\n\n"
        f"QUESTION: {question}"
    )
    return _PLAN_SYSTEM, user


def plan_query(question: str, df: pd.DataFrame) -> Tuple[Optional[Dict[str, Any]], str]:
    """Ask the LLM for an operation spec. Returns (plan_or_None, raw_response)."""
    system, user = build_plan_prompt(question, df)
    raw = llm_client.complete_text(system, user, max_tokens=800)
    parsed = schema_mod._parse_json_lenient(raw)
    if not isinstance(parsed, dict):
        return None, raw
    return parsed, raw


def execute_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> StatResult:
    """Run a validated operation spec against the dataset. Deterministic."""
    op = plan.get("operation")
    filters = plan.get("filters") or None
    if op == "frequency":
        return frequency(df, plan.get("field"), filters=filters)
    if op == "crosstab":
        return crosstab(
            df,
            plan.get("row_field"),
            plan.get("col_field"),
            percent=plan.get("percent") or None,
            filters=filters,
        )
    if op == "numeric_summary":
        return numeric_summary(
            df, plan.get("field"), group_by=plan.get("group_by") or None, filters=filters
        )
    if op == "compare_subsets":
        return compare_subsets(df, plan.get("metric"), plan.get("by"), filters=filters)
    if op == "unsupported":
        raise StatsError(
            f"Not answerable statistically from this dataset: "
            f"{plan.get('reason') or 'no reason given'}"
        )
    raise StatsError(f"Unknown operation {op!r} in plan.")


def narrate_result(question: str, result: StatResult) -> str:
    """One short plain-English paragraph about an already-computed table.

    The computed numbers are supplied in the prompt; the model is forbidden from
    introducing any figure not present in them.
    """
    system = (
        "You write the findings sentence(s) for a statistical table from an E&S "
        "survey, in a tone suitable for an ESIA/lender report.\n"
        "Rules:\n"
        "- Use ONLY numbers that appear in the provided table/metadata. Never "
        "derive, extrapolate, or invent figures.\n"
        "- State the sample size.\n"
        "- 2-4 sentences, plain English, no markdown."
    )
    user = (
        f"QUESTION ASKED: {question}\n\n"
        f"TITLE: {result.title}\n"
        f"ROWS ANALYSED (n): {result.n}\n"
        f"METHODOLOGY: {result.methodology}\n"
        f"CAVEATS: {'; '.join(result.caveats) if result.caveats else '(none)'}\n\n"
        f"COMPUTED TABLE:\n{result.table_text()}"
    )
    return llm_client.complete_text(system, user, max_tokens=500)


def compute_headline_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Every number a narrative summary is allowed to mention. Plain pandas —
    no LLM involved in computing any of these."""
    n = len(df)
    headline: Dict[str, Any] = {"n_households": n}

    def _pct(mask) -> float:
        return round(float(mask.mean()) * 100, 1) if n else 0.0

    if "vulnerable_household" in df.columns:
        headline["pct_vulnerable"] = _pct(df["vulnerable_household"] == "Yes")
    if "female_headed" in df.columns:
        headline["pct_female_headed"] = _pct(df["female_headed"] == "Yes")
    if "improved_water" in df.columns:
        headline["pct_unimproved_water"] = _pct(df["improved_water"] == "No")
    if "electricity_access" in df.columns:
        headline["pct_no_electricity"] = _pct(df["electricity_access"] == "No")
    if "tenure_category" in df.columns:
        headline["tenure_breakdown"] = df["tenure_category"].value_counts().to_dict()
    if "monthly_income_usd" in df.columns:
        inc = pd.to_numeric(df["monthly_income_usd"], errors="coerce").dropna()
        if len(inc):
            headline["mean_income_usd"] = round(float(inc.mean()), 2)
            headline["median_income_usd"] = round(float(inc.median()), 2)
    if "household_size" in df.columns:
        size = pd.to_numeric(df["household_size"], errors="coerce").dropna()
        if len(size):
            headline["mean_household_size"] = round(float(size.mean()), 2)
    if "food_security_category" in df.columns:
        headline["food_security_breakdown"] = df["food_security_category"].value_counts().to_dict()
    if "village" in df.columns:
        headline["n_villages"] = int(df["village"].nunique())
    if {"monthly_income_usd", "village"} <= set(df.columns):
        try:
            cmp_ = compare_subsets(df, "monthly_income_usd", "village")
            headline["highest_income_village"] = str(cmp_.table.iloc[0]["village"])
            headline["lowest_income_village"] = str(cmp_.table.iloc[-1]["village"])
        except StatsError:
            pass
    return headline


_NARRATIVE_SYSTEM = (
    "You write a socioeconomic profile narrative for a Resettlement Action Plan "
    "(RAP) or ESIA/ESDD chapter, from already-computed survey statistics.\n"
    "Structure it as short paragraphs by topic, using only the topics for which "
    "data is actually supplied: household composition & demographics; "
    "livelihoods & income; vulnerability; housing, land tenure & basic "
    "services; and community concerns.\n"
    "Rules:\n"
    "- Use ONLY the figures in HEADLINE STATS and the themes in CONCERN THEMES. "
    "Never derive, extrapolate, round differently, or invent a number or claim "
    "not supported by them.\n"
    "- State the sample size (n_households) in the opening paragraph.\n"
    "- Plain English, professional tone suitable for direct inclusion in an IFI "
    "submission. Flowing paragraphs only — no bullet points, no markdown "
    "headers, no section numbers.\n"
    "- Omit a topic entirely rather than guessing if no supporting data is given."
)


def write_narrative_summary(
    df: pd.DataFrame, *, theme_summaries: Optional[List[Any]] = None
) -> str:
    """A grounded, multi-paragraph socioeconomic-profile narrative — the
    'narrative_summary' plan operation, and reused by stats_report.py's
    executive summary so both surfaces produce the same quality of write-up.

    If `theme_summaries` isn't supplied and the dataset has free-text
    responses, this computes them itself (one extra qualitative-synthesis
    call) — pass already-computed summaries in to avoid that when the caller
    (e.g. the full report) already has them.
    """
    headline = compute_headline_stats(df)
    if theme_summaries is None and "project_concern" in df.columns:
        import qualitative  # local import: avoids a hard top-level dependency

        try:
            tags = qualitative.tag_responses(df)
            theme_summaries = qualitative.summarise_themes(tags)
        except llm_client.LLMError:
            theme_summaries = []
    themes_brief = [
        {"theme": s.theme, "count": s.count, "pct": s.pct} for s in (theme_summaries or [])[:6]
    ]
    user = (
        f"HEADLINE STATS:\n{json.dumps(headline, indent=2)}\n\n"
        f"CONCERN THEMES (ranked):\n{json.dumps(themes_brief, indent=2)}"
    )
    return llm_client.complete_text(_NARRATIVE_SYSTEM, user, max_tokens=1200)


def answer_statistical_question(
    question: str, df: pd.DataFrame
) -> Tuple[StatResult, str, Dict[str, Any]]:
    """Plan -> execute -> narrate. Returns (result, narration, plan).

    A 'narrative_summary' plan is handled separately: the "result" is a
    one-row headline-stats table (so the existing table+narration UI still
    renders something), and the "narration" is the full grounded write-up.
    """
    plan, raw = plan_query(question, df)
    if plan is None:
        raise StatsError(f"Could not parse an operation plan from the model: {raw[:300]}")
    if plan.get("operation") == "narrative_summary":
        headline = compute_headline_stats(df)
        result = StatResult(
            title="Headline statistics",
            table=pd.DataFrame([headline]),
            n=len(df),
            methodology="Summary figures computed across the full dataset; see narrative below.",
        )
        narration = write_narrative_summary(df)
        return result, narration, plan
    result = execute_plan(df, plan)
    narration = narrate_result(question, result)
    return result, narration, plan
