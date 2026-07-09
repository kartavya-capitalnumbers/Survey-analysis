"""
stats_app.py — Survey Statistical Analysis (Mode 1) — Streamlit prototype.

Run:
    streamlit run stats_app.py

Companion to app.py (the narrative Mode 2 prototype) — deliberately a SEPARATE
entry point so the two demos never interfere. This one covers the spec's
Survey Data Analysis feature end to end for one form type (household
socioeconomic surveys):
  - Ingestion: upload the standardised PDF survey FORMS themselves —
    household_extraction.py reads each one (digital text, OCR fallback for
    scans, booklet-splitting for multi-form PDFs) and maps it to a structured
    dataset row with one Bedrock call per form, no manual template
    configuration.
  - Statistical outputs (stats.py + demographics.py): frequencies, cross-tabs,
    numeric summaries, subset comparisons, and a canned demographic-breakdown
    battery (gender / age group / vulnerability / zone) — every table
    labelled with sample size, methodology, and confidence flags; CSV + Excel
    export.
  - Qualitative outputs (qualitative.py): thematic synthesis, sentiment,
    ranked concerns, anonymised representative quotes over the survey's
    free-text responses.
  - Charts (charts.py): bar / pie / demographic-pyramid, PNG export.
  - Narrative + FAQ batch report (stats_report.py): one operation produces a
    complete DOCX — executive summary, demographics, charts, qualitative
    synthesis — the LLM only narrates already-computed figures, never invents.
  - Plain-English analysis questions: the LLM only PLANS the operation and
    NARRATES the computed table; it never produces a number itself.
Auth/config is identical to app.py (default AWS credential chain; model/region
in the sidebar via llm_client).
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

import charts
import demographics
import household_extraction
import llm_client
import qualitative
import stats
import stats_report


st.set_page_config(page_title="Survey Statistical Analysis", layout="wide")


# ---------------------------------------------------------------------------
# The 10 IFC PS5 / RAP question categories are baked into stats._PLAN_SYSTEM
# (the planner prompt) so the user can ask their own question in any of these
# areas and have it routed correctly — see _DOMAIN_CATEGORIES in stats.py.
# No fixed question list here; the "Ask in plain English" tab is the surface.
# ---------------------------------------------------------------------------
FAQ_CATEGORY_NAMES = [
    "Census & demographics", "Land tenure & displacement", "Vulnerability identification",
    "Livelihoods & income", "Housing & infrastructure", "Food security",
    "Resettlement preferences & concerns", "Cross-tabulations & equity analysis",
    "Qualitative synthesis", "Compliance & reporting (IFC PS5 completeness)",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    st.session_state.setdefault("survey_df", None)      # pandas DataFrame
    st.session_state.setdefault("survey_name", None)    # source file name
    st.session_state.setdefault("stat_history", [])     # list[(question, plan, StatResult, narration)]
    st.session_state.setdefault("extract_members_df", None)  # de-identified member roster, if extracted from PDFs
    st.session_state.setdefault("extract_raw_pii_df", None)  # gated full-PII export, if extracted from PDFs
    st.session_state.setdefault("extract_report", None)      # list[(filename, "ok"/"failed", detail)]
    st.session_state.setdefault("qual_tags", None)            # list[qualitative.ResponseTag], cached
    st.session_state.setdefault("qual_themes", None)          # list[qualitative.ThemeSummary], cached
    st.session_state.setdefault("stats_report_bytes", None)   # cached generated DOCX


def _stem() -> str:
    name = st.session_state.survey_name or "survey"
    return os.path.splitext(name)[0]


def _clear_dataset_state() -> None:
    st.session_state.stat_history = []
    st.session_state.qual_tags = None
    st.session_state.qual_themes = None
    st.session_state.stats_report_bytes = None


def _render_result(result: stats.StatResult, *, key_prefix: str = "") -> None:
    st.markdown(f"**{result.title}**")
    st.dataframe(result.table, width="stretch", hide_index=True)
    st.caption(f"n = {result.n} · {result.methodology}")
    for c in result.caveats:
        st.caption(f"⚠️ {c}")
    safe_title = "".join(ch if ch.isalnum() else "_" for ch in result.title)[:40]
    key = f"{key_prefix}{safe_title}_{result.n}_{id(result)}"
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "⬇️ CSV", data=result.to_csv(), file_name=_stem() + f"_{safe_title}.csv",
            mime="text/csv", key=f"csv_{key}",
        )
    with c2:
        st.download_button(
            "⬇️ Excel (with confidence flags)", data=result.to_excel_bytes(),
            file_name=_stem() + f"_{safe_title}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_{key}",
        )


def _render_chart_with_png(chart: alt.Chart, *, name: str) -> None:
    st.altair_chart(chart, width="stretch")
    st.download_button(
        "⬇️ Download chart (PNG)",
        data=charts.chart_to_png(chart),
        file_name=f"{_stem()}_{name}.png",
        mime="image/png",
        key=f"png_{name}",
    )


_init_state()


# ---------------------------------------------------------------------------
# Sidebar — config (same pattern as app.py)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    model_input = st.text_input("Bedrock model id", value=llm_client.active_model())
    region_input = st.text_input("AWS region", value=llm_client.active_region())
    llm_client.set_credentials(model=model_input or None, region=region_input or None)
    creds_ok = llm_client.has_credentials()
    st.write("**AWS credentials:**", "✅ default chain" if creds_ok else "❌ none found")
    st.divider()
    st.caption(
        "Prototype • PDF survey forms • deterministic pandas statistics "
        "(LLM plans & narrates only — it never computes numbers)."
    )


st.title("📊 Survey Statistical Analysis")
st.caption("Mode 1 (survey data analysis) — Phase 1 statistics prototype")


# ---------------------------------------------------------------------------
# 1. Dataset ingestion — upload survey form PDFs; each becomes one row
# ---------------------------------------------------------------------------

st.subheader("📄 Upload survey forms (PDF)")
st.write(
    "Upload scanned/typed household survey forms. Each PDF is read (digital "
    "text, with OCR fallback for scans) and mapped to a structured row by "
    "one Bedrock call per form — no manual template configuration. A single "
    "PDF containing several forms back to back (a scanned booklet) is "
    "detected and split automatically."
)
pdf_files = st.file_uploader(
    "Survey form PDFs (one form per file, or a multi-form booklet)",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader",
)
extract_clicked = st.button(
    "🔎 Extract & build dataset", type="primary", disabled=not pdf_files
)

if extract_clicked and pdf_files:
    raw_rows, analysis_rows, member_rows, report = [], [], [], []
    with st.status(f"Extracting {len(pdf_files)} file(s)…", expanded=True) as status:
        for f in pdf_files:
            st.write(f"Reading **{f.name}**…")
            for r in household_extraction.extract_forms(f.getvalue(), f.name):
                if not r.ok:
                    report.append((r.label, "failed", r.error))
                    st.write(f"⚠️ {r.label}: {r.error}")
                    continue
                raw_rows.append(household_extraction.raw_row(r.form))
                analysis_rows.append(household_extraction.analysis_row(r.form))
                member_rows.extend(household_extraction.member_rows(r.form))
                report.append(
                    (r.label, "ok", r.form.record.get("household_id") or "(no ID found)")
                )

        n_ok = sum(1 for _, s, _ in report if s == "ok")
        status.update(
            label=f"Extraction complete — {n_ok}/{len(report)} form(s) parsed ✅",
            state="complete",
        )

    st.session_state.extract_report = report
    if analysis_rows:
        st.session_state.survey_df = pd.DataFrame(analysis_rows)
        st.session_state.survey_name = "extracted_survey_forms.csv"
        st.session_state.extract_members_df = pd.DataFrame(member_rows) if member_rows else None
        st.session_state.extract_raw_pii_df = pd.DataFrame(raw_rows)
        _clear_dataset_state()
    else:
        st.session_state.extract_members_df = None
        st.session_state.extract_raw_pii_df = None

if st.session_state.extract_report:
    n_ok = sum(1 for _, s, _ in st.session_state.extract_report if s == "ok")
    n_total = len(st.session_state.extract_report)
    with st.expander(
        f"Last extraction: {n_ok}/{n_total} form(s) parsed", expanded=(n_ok < n_total)
    ):
        for name, status_, detail in st.session_state.extract_report:
            icon = "✅" if status_ == "ok" else "⚠️"
            st.write(f"{icon} **{name}** — {detail}")


df = st.session_state.survey_df
if df is not None:
    st.divider()
    st.subheader(f"Dataset — {st.session_state.survey_name}")
    profile = stats.profile_columns(df)
    st.caption(f"{len(df)} rows × {len(df.columns)} columns")
    with st.expander("Column profile", expanded=False):
        st.dataframe(pd.DataFrame(profile), width="stretch", hide_index=True)
    with st.expander("Preview (first 20 rows)", expanded=False):
        st.dataframe(df.head(20), width="stretch")

    members_df = st.session_state.extract_members_df
    if members_df is not None:
        with st.expander(f"Household members — {len(members_df)} individuals (de-identified)"):
            st.dataframe(members_df, width="stretch", hide_index=True)

    tab_quick, tab_demo, tab_qual, tab_report, tab_nl = st.tabs(
        [
            "📊 Quick statistics",
            "👥 Demographics",
            "💭 Qualitative synthesis",
            "🧾 Standard report (FAQ mode)",
            "💬 Ask in plain English",
        ]
    )

    # --- Quick statistics (no LLM involved at all) ---------------------------
    with tab_quick:
        cat_cols = [p["name"] for p in profile if p["kind"] in ("categorical", "boolean")]
        num_cols = [p["name"] for p in profile if p["kind"] == "numeric"]
        all_cols = [p["name"] for p in profile]

        op = st.radio(
            "Operation",
            ["Frequency (counts + %)", "Cross-tabulation", "Numeric summary", "Compare subsets"],
            horizontal=True,
        )
        try:
            if op == "Frequency (counts + %)":
                fld = st.selectbox("Field", cat_cols or all_cols)
                if st.button("Compute", key="go_freq"):
                    result = stats.frequency(df, fld)
                    _render_result(result, key_prefix="freq_")
                    _render_chart_with_png(
                        charts.bar_chart(result.table, fld, "count", title=result.title),
                        name=f"freq_{fld}",
                    )
            elif op == "Cross-tabulation":
                c1, c2, c3 = st.columns(3)
                with c1:
                    row_f = st.selectbox("Rows", cat_cols or all_cols, index=0)
                with c2:
                    col_f = st.selectbox("Columns", cat_cols or all_cols, index=min(1, len(all_cols) - 1))
                with c3:
                    pct = st.selectbox("Percentages", ["none", "row", "col", "overall"])
                if st.button("Compute", key="go_ct"):
                    _render_result(
                        stats.crosstab(df, row_f, col_f, percent=None if pct == "none" else pct),
                        key_prefix="ct_",
                    )
            elif op == "Numeric summary":
                c1, c2 = st.columns(2)
                with c1:
                    fld = st.selectbox("Numeric field", num_cols or all_cols)
                with c2:
                    grp = st.selectbox("Group by (optional)", ["(none)"] + cat_cols)
                if st.button("Compute", key="go_num"):
                    result = stats.numeric_summary(df, fld, group_by=None if grp == "(none)" else grp)
                    _render_result(result, key_prefix="num_")
                    if grp != "(none)":
                        _render_chart_with_png(
                            charts.bar_chart(result.table, grp, "mean", title=result.title),
                            name=f"num_{fld}_by_{grp}",
                        )
            else:  # Compare subsets
                c1, c2 = st.columns(2)
                with c1:
                    metric = st.selectbox("Metric (numeric)", num_cols or all_cols, key="cmp_metric")
                with c2:
                    by = st.selectbox("Compare across", cat_cols or all_cols, key="cmp_by")
                if st.button("Compute", key="go_cmp"):
                    result = stats.compare_subsets(df, metric, by)
                    _render_result(result, key_prefix="cmp_")
                    _render_chart_with_png(
                        charts.bar_chart(result.table, by, "mean", title=result.title),
                        name=f"cmp_{metric}_by_{by}",
                    )
        except stats.StatsError as exc:
            st.error(str(exc))

    # --- Demographics: canned breakdowns by gender/age/vulnerability/zone ----
    with tab_demo:
        st.write(
            "Standard demographic breakdowns — gender, age group, vulnerability "
            "category, and geographic zone — computed with the same "
            "deterministic engine as Quick statistics."
        )
        demo_results = demographics.run_breakdowns(df)
        for i, result in enumerate(demo_results):
            with st.container(border=True):
                _render_result(result, key_prefix=f"demo{i}_")
                # Only frequency tables (category, count, percent) chart cleanly
                # as a simple bar — crosstabs and grouped summaries need a
                # different shape of chart than this module builds.
                is_frequency = "count" in result.table.columns and "percent" in result.table.columns
                if is_frequency:
                    cat_col = result.table.columns[0]
                    _render_chart_with_png(
                        charts.bar_chart(result.table, cat_col, "count", title=result.title),
                        name=f"demo_{i}",
                    )

        if members_df is not None and not members_df.empty:
            st.markdown("**Population pyramid** (age group × sex, all extracted members)")
            _render_chart_with_png(charts.demographic_pyramid(members_df), name="pyramid")

    # --- Qualitative synthesis: thematic / sentiment / ranked / anonymised ---
    with tab_qual:
        st.write(
            "Classifies every free-text 'project concern' response into themes "
            "and sentiment, then ranks themes and surfaces anonymised "
            "representative quotes. One batch LLM call per ~40 responses; "
            "counting and ranking are pure Python."
        )
        has_text = "project_concern" in df.columns and df["project_concern"].astype(str).str.strip().any()
        if not has_text:
            st.info("No free-text 'project_concern' responses found in this dataset.")
        else:
            if st.button("💭 Run thematic & sentiment synthesis", disabled=st.session_state.qual_tags is not None):
                try:
                    with st.spinner("Classifying responses…"):
                        tags = qualitative.tag_responses(df)
                        themes = qualitative.summarise_themes(tags)
                    st.session_state.qual_tags = tags
                    st.session_state.qual_themes = themes
                except llm_client.LLMError as exc:
                    st.error(f"LLM error: {exc}")

            if st.session_state.qual_tags is not None:
                if st.button("🔄 Re-run synthesis"):
                    st.session_state.qual_tags = None
                    st.session_state.qual_themes = None
                    st.rerun()

                tags = st.session_state.qual_tags
                themes = st.session_state.qual_themes
                st.caption(f"{len(tags)} responses classified into {len(themes)} themes.")

                st.markdown("**Ranked concern themes**")
                themes_tab = qualitative.themes_table(themes)
                st.dataframe(themes_tab, width="stretch", hide_index=True)
                st.download_button(
                    "⬇️ Download themes (CSV)", data=themes_tab.to_csv(index=False),
                    file_name=_stem() + "_themes.csv", mime="text/csv",
                )

                st.markdown("**Sentiment breakdown**")
                sent_tab = qualitative.sentiment_table(tags)
                st.dataframe(sent_tab, width="stretch", hide_index=True)
                _render_chart_with_png(
                    charts.pie_chart(sent_tab, "sentiment", "count", title="Sentiment"),
                    name="sentiment_pie",
                )

                st.markdown("**Representative quotes** (anonymised)")
                for s in themes:
                    with st.container(border=True):
                        st.markdown(f"**{s.theme}** — {s.count} response(s), {s.pct}%")
                        for q in s.sample_quotes:
                            st.markdown(f"> {q}")

    # --- Standard report (FAQ mode): survey-type selector + one-click report -
    with tab_report:
        st.write(
            "Pre-built standard question sets, run as one operation. Select a "
            "survey type and generate the complete analytical report — "
            "demographics, charts, and qualitative synthesis included."
        )
        survey_type = st.selectbox(
            "Survey type",
            [
                "Household socioeconomic",
                "Biodiversity baseline (not available)",
                "Grievance intake (not available)",
            ],
        )
        if survey_type != "Household socioeconomic":
            st.caption(
                "⚠️ This prototype only implements extraction and analysis for "
                "household socioeconomic surveys. Biodiversity baseline and "
                "grievance intake would need their own extraction schema "
                "(see household_extraction.py) before this mode could run."
            )
        run_report = st.button(
            "🚀 Run standard analysis suite", type="primary",
            disabled=survey_type != "Household socioeconomic",
        )
        if run_report:
            try:
                with st.status("Generating report…", expanded=True) as status:
                    st.write("Computing demographic breakdowns…")
                    st.write("Classifying free-text responses (thematic + sentiment)…")
                    st.write("Writing executive summary…")
                    st.write("Building charts and assembling the document…")
                    docx_bytes = stats_report.make_stats_report_docx(
                        st.session_state.survey_name, df, members_df
                    )
                    status.update(label="Report complete ✅", state="complete")
                st.session_state.stats_report_bytes = docx_bytes
            except llm_client.LLMError as exc:
                st.error(f"LLM error: {exc}")

        if st.session_state.stats_report_bytes is not None:
            st.success("Report ready.")
            st.download_button(
                "⬇️ Download standard analysis report (DOCX)",
                data=st.session_state.stats_report_bytes,
                file_name=_stem() + "_standard_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

    # --- Plain-English questions ---------------------------------------------
    with tab_nl:
        st.write(
            "Ask an analysis question. The model plans the statistic; pandas "
            "computes it; the model then narrates the computed numbers."
        )
        st.caption("Ask about: " + " · ".join(FAQ_CATEGORY_NAMES))
        examples = [
            "What percentage of households are female-headed?",
            "What is the sex disaggregation of the surveyed population?",
            "Which households report the most severe food insecurity and what are their IDs?",
            "Compare monthly income across land tenure categories",
            "Does the dataset include sex-disaggregated data for all key indicators?",
        ]
        st.caption("Examples: " + " · ".join(f"_{e}_" for e in examples))
        question = st.text_input("Your question", key="stat_q")
        if st.button("Analyse", disabled=not question):
            try:
                with st.spinner("Planning and computing…"):
                    result, narration, plan = stats.answer_statistical_question(
                        question, df, members_df=members_df
                    )
                st.session_state.stat_history.insert(0, (question, plan, result, narration))
            except stats.StatsError as exc:
                st.error(str(exc))
            except llm_client.LLMError as exc:
                st.error(f"LLM error: {exc}")

        for i, (q, plan, result, narration) in enumerate(st.session_state.stat_history):
            with st.container(border=True):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**Finding:** {narration}")
                _render_result(result, key_prefix=f"nl{i}_")
                with st.expander("Operation plan (audit trail)"):
                    st.json(plan)

    st.divider()
    dl_col, pii_col = st.columns([1, 1])
    with dl_col:
        st.download_button(
            "⬇️ Download analysis dataset (CSV, de-identified)",
            data=df.to_csv(index=False),
            file_name=_stem() + "_dataset.csv",
            mime="text/csv",
        )
        if members_df is not None:
            st.download_button(
                "⬇️ Download household members (CSV, de-identified)",
                data=members_df.to_csv(index=False),
                file_name=_stem() + "_members.csv",
                mime="text/csv",
            )
    with pii_col:
        raw_pii_df = st.session_state.extract_raw_pii_df
        if raw_pii_df is not None:
            st.caption(
                "⚠️ Contains names, national IDs, phone numbers and GPS. "
                "Export only when explicitly needed."
            )
            st.download_button(
                "⬇️ Download raw transcription — FULL PII",
                data=raw_pii_df.to_csv(index=False),
                file_name=_stem() + "_raw_FULL_PII.csv",
                mime="text/csv",
            )
