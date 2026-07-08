"""
qualitative.py — thematic synthesis, sentiment, ranked concerns, and anonymised
quote extraction over the survey's free-text responses (project_concern).

Design rule (same grounding pattern as stats.py): the LLM classifies and
redacts EACH response individually; it never counts, ranks, or aggregates —
all of that is plain Python over the tags it returns. One batch call handles
many responses at once (consistent theme labels across the batch, and far
cheaper than one call per response); large batches are chunked so no single
call risks an oversized request.

Anonymisation is belt-and-suspenders: the LLM is instructed to redact names/
phone numbers/exact ages from each quote, and `tag_responses()` additionally
strips any known respondent name from the model's output as a deterministic
safety net — the caller supplies that name lookup (typically built from the
gated raw-PII export), so this module itself never needs to see or export PII.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

import llm_client
import schema as schema_mod

_BATCH_SIZE = 40  # responses per LLM call — keeps each call small and reliable


@dataclass
class ResponseTag:
    household_id: str
    theme: str
    sentiment: str  # positive | neutral | negative
    anonymised_quote: str


@dataclass
class ThemeSummary:
    theme: str
    count: int
    pct: float
    sentiment_breakdown: Dict[str, int]
    sample_quotes: List[str]


_SYSTEM = (
    "You analyse free-text responses from an E&S/resettlement household survey. "
    "For EACH numbered response, return:\n"
    "  - theme: a short (2-4 word) theme label for its main concern/topic. Use "
    "a SMALL, CONSISTENT set of themes across all responses — reuse the same "
    "label for similar concerns rather than inventing near-duplicate labels.\n"
    "  - sentiment: exactly one of positive, neutral, negative — the "
    "respondent's tone toward the project/situation.\n"
    "  - anonymised_quote: the response rewritten to preserve its substance "
    "while removing personal names, phone numbers, exact ages, or other "
    "directly identifying details (replace with '[Respondent]' or a generic "
    "descriptor). Keep it a genuine quote-like sentence, not a paraphrase that "
    "changes the meaning.\n\n"
    "Rules:\n"
    "- Classify and redact ONLY — never invent a response, never merge or "
    "drop items. Return exactly one result per input, in the same order.\n"
    "- Return ONLY a JSON array: "
    '[{"theme": "...", "sentiment": "...", "anonymised_quote": "..."}, ...]. '
    "No prose, no markdown fences."
)


def build_prompt(responses: List[str]) -> Tuple[str, str]:
    numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(responses))
    return _SYSTEM, f"RESPONSES ({len(responses)} total):\n{numbered}"


def _redact_names(text: str, names: List[str]) -> str:
    """Deterministic safety net: strip any known respondent name that survived
    the model's own anonymisation pass."""
    for name in names:
        for part in (name or "").split():
            if len(part) > 2:
                text = re.sub(re.escape(part), "[Respondent]", text, flags=re.IGNORECASE)
    return text


def _tag_batch(responses: List[str]) -> List[dict]:
    system, user = build_prompt(responses)
    raw = llm_client.complete_text(system, user, max_tokens=4000)
    parsed = schema_mod._parse_json_lenient(raw)
    if not isinstance(parsed, list) or len(parsed) != len(responses):
        got = len(parsed) if isinstance(parsed, list) else "invalid JSON"
        raise llm_client.LLMError(
            f"Thematic synthesis returned {got} tags for {len(responses)} "
            f"responses in this batch — could not align results."
        )
    return parsed


def tag_responses(
    df: pd.DataFrame,
    text_col: str = "project_concern",
    *,
    id_col: str = "household_id",
    known_names: Optional[Dict[str, str]] = None,
) -> List[ResponseTag]:
    """Classify + anonymise every non-empty response in `text_col`.

    Returns one ResponseTag per non-empty response. Batched (see
    `_BATCH_SIZE`) so a large survey doesn't risk one oversized call.
    """
    sub = df[[id_col, text_col]].copy()
    sub[text_col] = sub[text_col].astype(str).str.strip()
    sub = sub[sub[text_col].str.len() > 0].reset_index(drop=True)
    if sub.empty:
        return []

    known_names = known_names or {}
    tags: List[ResponseTag] = []
    for start in range(0, len(sub), _BATCH_SIZE):
        chunk = sub.iloc[start : start + _BATCH_SIZE]
        results = _tag_batch(chunk[text_col].tolist())
        for (_, row), tag in zip(chunk.iterrows(), results):
            hh_id = str(row[id_col])
            quote = _redact_names(
                str(tag.get("anonymised_quote", "")).strip(), [known_names.get(hh_id, "")]
            )
            tags.append(
                ResponseTag(
                    household_id=hh_id,
                    theme=str(tag.get("theme", "Uncategorised")).strip() or "Uncategorised",
                    sentiment=str(tag.get("sentiment", "neutral")).strip().lower(),
                    anonymised_quote=quote,
                )
            )
    return tags


def summarise_themes(tags: List[ResponseTag], *, max_quotes: int = 3) -> List[ThemeSummary]:
    """Deterministic aggregation: count and rank themes. No LLM call here —
    every number comes from counting the tags already classified above."""
    if not tags:
        return []
    total = len(tags)
    by_theme: Dict[str, List[ResponseTag]] = {}
    for t in tags:
        by_theme.setdefault(t.theme, []).append(t)

    summaries = []
    for theme, items in by_theme.items():
        sentiment_counts: Dict[str, int] = {}
        for it in items:
            sentiment_counts[it.sentiment] = sentiment_counts.get(it.sentiment, 0) + 1
        summaries.append(
            ThemeSummary(
                theme=theme,
                count=len(items),
                pct=round(len(items) / total * 100, 1),
                sentiment_breakdown=sentiment_counts,
                sample_quotes=[it.anonymised_quote for it in items[:max_quotes]],
            )
        )
    summaries.sort(key=lambda s: s.count, reverse=True)
    return summaries


def sentiment_table(tags: List[ResponseTag]) -> pd.DataFrame:
    """Sentiment frequency as a plain DataFrame (counting only — no LLM)."""
    if not tags:
        return pd.DataFrame(columns=["sentiment", "count", "percent"])
    s = pd.Series([t.sentiment for t in tags])
    counts = s.value_counts()
    total = len(s)
    return pd.DataFrame(
        {
            "sentiment": counts.index,
            "count": counts.to_numpy(),
            "percent": (counts.to_numpy() / total * 100).round(1),
        }
    )


def themes_table(summaries: List[ThemeSummary]) -> pd.DataFrame:
    """Ranked concern table for display/export — count + % + sentiment split."""
    rows = []
    for s in summaries:
        rows.append(
            {
                "theme": s.theme,
                "count": s.count,
                "percent": s.pct,
                "positive": s.sentiment_breakdown.get("positive", 0),
                "neutral": s.sentiment_breakdown.get("neutral", 0),
                "negative": s.sentiment_breakdown.get("negative", 0),
            }
        )
    return pd.DataFrame(rows)
