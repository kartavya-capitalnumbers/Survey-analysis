"""
charts.py — bar / pie / demographic-pyramid charts over survey statistics.

Per the spec's Phase 2 "Charts" output: bar charts, pie charts, and
demographic pyramids auto-generated from statistical outputs, exportable as
PNG or embedded in report documents. Built with Altair (Streamlit's preferred
charting library) so `st.altair_chart` renders them theme-aware (light/dark)
automatically; `chart_to_png()` uses vl-convert to rasterise the same spec for
download / DOCX embedding, where Streamlit's live theming doesn't apply.

Every function takes an already-computed DataFrame/StatResult — no chart here
computes a number; they only encode numbers stats.py or demographics.py already
produced, matching the app's grounding rule.

Color: the categorical hue order below is a fixed 8-slot palette (validated for
colorblind-safe adjacent contrast) — series are assigned hues by POSITION in
this list, never re-cycled or re-ranked, so the same category always gets the
same color across charts.
"""

from __future__ import annotations

import io
from typing import List, Optional

import altair as alt
import pandas as pd

_CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
_BLUE = _CATEGORICAL[0]
_MAGENTA = _CATEGORICAL[6]


def _labelled_bar(
    df: pd.DataFrame, cat_col: str, val_col: str, *, color: str = _BLUE, horizontal: bool = True
) -> alt.Chart:
    base = alt.Chart(df)
    if horizontal:
        bars = base.mark_bar(color=color, cornerRadiusEnd=3).encode(
            y=alt.Y(f"{cat_col}:N", sort="-x", title=None),
            x=alt.X(f"{val_col}:Q", title=val_col),
            tooltip=[cat_col, val_col],
        )
        labels = base.mark_text(align="left", dx=4, color=color).encode(
            y=alt.Y(f"{cat_col}:N", sort="-x"),
            x=alt.X(f"{val_col}:Q"),
            text=alt.Text(f"{val_col}:Q"),
        )
    else:
        bars = base.mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X(f"{cat_col}:N", sort="-y", title=None),
            y=alt.Y(f"{val_col}:Q", title=val_col),
            tooltip=[cat_col, val_col],
        )
        labels = base.mark_text(align="center", dy=-6, color=color).encode(
            x=alt.X(f"{cat_col}:N", sort="-y"),
            y=alt.Y(f"{val_col}:Q"),
            text=alt.Text(f"{val_col}:Q"),
        )
    return (bars + labels).properties(
        height=alt.Step(28) if horizontal else 260, background="transparent"
    )


def bar_chart(
    df: pd.DataFrame, cat_col: str, val_col: str, *, title: str = "", horizontal: bool = True
) -> alt.Chart:
    """Single-series bar chart — one bar per category, direct value labels.

    Category identity is already carried by the axis labels, so all bars share
    one hue rather than a meaningless per-bar color.
    """
    return _labelled_bar(df, cat_col, val_col, horizontal=horizontal).properties(title=title)


def pie_chart(df: pd.DataFrame, cat_col: str, val_col: str, *, title: str = "") -> alt.Chart:
    """Pie/donut chart — one wedge per category, fixed categorical hue order,
    with a legend (color is the only identity cue here, so it must be present)
    and direct percentage labels on wedges.
    """
    total = df[val_col].sum()
    data = df.copy()
    data["_pct"] = (data[val_col] / total * 100).round(1) if total else 0.0
    cats = list(data[cat_col])
    colors = [_CATEGORICAL[i % len(_CATEGORICAL)] for i in range(len(cats))]

    base = alt.Chart(data).encode(
        theta=alt.Theta(f"{val_col}:Q", stack=True),
        color=alt.Color(
            f"{cat_col}:N",
            scale=alt.Scale(domain=cats, range=colors),
            legend=alt.Legend(title=None),
        ),
        tooltip=[cat_col, val_col, alt.Tooltip("_pct:Q", title="percent")],
    )
    wedges = base.mark_arc(innerRadius=60, stroke="white", strokeWidth=2)
    labels = base.mark_text(radius=110, size=12).encode(
        text=alt.Text("_pct:Q", format=".0f"),
        color=alt.value("#0b0b0b"),
    )
    return (wedges + labels).properties(title=title, background="transparent")


def demographic_pyramid(
    member_df: pd.DataFrame,
    *,
    age_group_col: str = "age_group",
    sex_col: str = "sex",
    age_order: Optional[List[str]] = None,
) -> alt.Chart:
    """Population pyramid — male bars extend left, female bars extend right,
    a shared age-group axis down the middle. Two-series, fixed colors + legend.
    """
    if age_order is None:
        age_order = ["0-14 (child)", "15-64 (working age)", "65+ (elderly)"]

    counts = (
        member_df.groupby([age_group_col, sex_col]).size().reset_index(name="count")
    )
    counts["signed"] = counts.apply(
        lambda r: -r["count"] if r[sex_col] == "Male" else r["count"], axis=1
    )
    max_abs = counts["count"].max() if len(counts) else 1

    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            y=alt.Y(f"{age_group_col}:N", sort=age_order, title=None),
            x=alt.X(
                "signed:Q",
                title="count",
                axis=alt.Axis(labelExpr="abs(datum.value)"),
                scale=alt.Scale(domain=[-max_abs * 1.15, max_abs * 1.15]),
            ),
            color=alt.Color(
                f"{sex_col}:N",
                scale=alt.Scale(domain=["Male", "Female"], range=[_BLUE, _MAGENTA]),
                legend=alt.Legend(title=None),
            ),
            tooltip=[age_group_col, sex_col, "count"],
        )
        .properties(background="transparent", height=alt.Step(36))
    )
    return chart


def chart_to_png(chart: alt.Chart, *, scale: float = 2.0) -> bytes:
    """Rasterise an Altair chart to PNG bytes (via vl-convert) for download or
    DOCX embedding, outside Streamlit's live theme context."""
    buf = io.BytesIO()
    chart.save(buf, format="png", scale_factor=scale)
    return buf.getvalue()
