"""
household_extraction.py — PDF household survey form -> structured dataset row.

This is the "Document Ingestion + OCR/Transcription + Form structure recognition"
half of the Survey Data Analysis spec, scoped to ONE standardised form type
(household socioeconomic survey) so it can plug straight into the statistics
engine (stats.py): one row per form, one column per survey field.

Pipeline per uploaded PDF:
    pdf bytes --(extraction.extract_pages, digital text + OCR fallback)--> full text
    full text --(split_into_forms: one chunk per form — handles a multi-form
        "booklet" PDF, the spec's other input shape, as well as the common
        one-form-per-PDF case)--> per-form text chunk(s)
    each chunk --(one Bedrock call, JSON schema population)--> raw record dict
    raw record --(normalise: type coercion, canonical categories, derived fields)-->
        one household row + its member rows

A chunk over `_MAX_CHARS_PER_FORM` characters is skipped with a clear error
instead of being sent to Bedrock, which would otherwise fail with an opaque
input-too-long error from the API.

PROTOTYPE NOTE
--------------
One LLM call maps free-form form text to named fields — the same "field mapping
without manual template configuration" mechanism as schema.py, applied to a
different (tabular, statistical) schema instead of a narrative one. Production
would route the scan through Textract first (per the spec's architecture notes)
and use this module's schema/normalisation/prompt as the Bedrock correction layer
on top of Textract's raw key-value output.

PII (name, national ID, phone, GPS) is extracted because the "raw transcription
export ... only where user explicitly requests full dataset export" requirement
means it must exist in ONE place — callers are responsible for keeping the raw
(PII) rows and the de-identified analysis rows in separate exports (stats_app.py
does this by gating the PII download behind an explicit, clearly-labelled button).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import extraction
import llm_client
import schema as schema_mod


# A form chunk longer than this is skipped rather than sent to Bedrock. This is
# a generous sanity ceiling (a real form runs 2,000-4,000 characters), not the
# model's actual context limit — it exists to fail fast and clearly on a
# garbled OCR dump or a mis-split chunk instead of surfacing Bedrock's opaque
# input-too-long error.
_MAX_CHARS_PER_FORM = 60_000

# A repeating per-form identifier line is the form-boundary signal for
# splitting a multi-form "booklet" PDF (the spec's other document-ingestion
# shape) into one chunk per household. This is a heuristic scoped to this
# module's one form type, not a general document-segmentation model — it
# looks for the identifier label every instance of THIS form repeats.
_FORM_BOUNDARY_RE = re.compile(
    r"(?im)^[ \t]*(?:household|hh|respondent|form)[\s_-]*id[\s_-]*(?:no\.?)?\s*[:#-]\s*\S+"
)


# ---------------------------------------------------------------------------
# Schema template — one entry per field this module extracts from a form.
# ---------------------------------------------------------------------------

_MEMBER_TEMPLATE: Dict[str, Any] = {
    "name": "", "age": "", "sex": "", "relationship": "", "education": "", "occupation": "",
}

HOUSEHOLD_TEMPLATE: Dict[str, Any] = {
    "household_id": "", "village": "", "district": "", "gps": "",
    "enumerator": "", "survey_date": "",
    "head_name": "", "national_id": "", "phone": "",
    "head_age": "", "head_gender": "", "education_level": "", "head_occupation": "",
    "ethnicity": "", "religion": "", "tenure_category": "", "household_size": "",
    "wall_type": "", "roof_type": "", "rooms": "", "floor_area_sqm": "",
    "water_source_type": "", "sanitation_type": "", "electricity_access": "",
    "primary_cooking_fuel": "", "land_area_ha": "", "livestock": "",
    "income_low": "", "income_high": "", "food_insecure_months": "",
    "vuln_elderly": False, "vuln_female_headed": False, "vuln_disabled": False,
    "vuln_chronic_illness": False, "vuln_minority": False,
    "project_concern": "",
    "members": [_MEMBER_TEMPLATE],
}

_VULN_FLAGS = [
    "vuln_elderly", "vuln_female_headed", "vuln_disabled",
    "vuln_chronic_illness", "vuln_minority",
]

_SCALAR_STR_FIELDS = [
    "household_id", "village", "district", "gps", "enumerator", "survey_date",
    "head_name", "national_id", "phone", "head_gender", "education_level",
    "head_occupation", "ethnicity", "religion", "tenure_category",
    "wall_type", "roof_type", "water_source_type", "sanitation_type",
    "electricity_access", "primary_cooking_fuel", "livestock", "project_concern",
]
_SCALAR_NUM_FIELDS = [
    "head_age", "household_size", "rooms", "floor_area_sqm", "land_area_ha",
    "income_low", "income_high", "food_insecure_months",
]

_RAW_ORDER = [
    "source_file", "household_id", "village", "district", "gps", "enumerator", "survey_date",
    "head_name", "national_id", "phone",
    "head_age", "head_gender", "education_level", "head_occupation",
    "ethnicity", "religion", "tenure_category", "household_size",
    "wall_type", "roof_type", "rooms", "floor_area_sqm",
    "water_source_type", "sanitation_type", "electricity_access", "primary_cooking_fuel",
    "land_area_ha", "livestock", "income_low", "income_high",
    "food_insecure_months", *_VULN_FLAGS, "project_concern",
]


# ---------------------------------------------------------------------------
# Derivation rules for the analysis-ready columns.
# ---------------------------------------------------------------------------

def improved_water(source: str) -> str:
    """JMP-style: a borehole/piped source is improved; well/river/spring are not."""
    return "Yes" if (source or "").strip().lower() in ("borehole", "piped") else "No"


def food_security_category(months: Optional[float]) -> str:
    """0 = food secure; 1-3 = moderately insecure; 4+ = highly insecure."""
    if months is None:
        return "Unknown"
    if months == 0:
        return "Food secure"
    if months <= 3:
        return "Moderately insecure"
    return "Highly insecure"


def size_band(size: Optional[float]) -> str:
    """Small 1-4; Medium 5-7; Large 8+."""
    if size is None:
        return "Unknown"
    if size <= 4:
        return "Small (1-4)"
    if size <= 7:
        return "Medium (5-7)"
    return "Large (8+)"


def age_group(age: Optional[float]) -> str:
    if age is None:
        return "Unknown"
    if age < 15:
        return "0-14 (child)"
    if age < 65:
        return "15-64 (working age)"
    return "65+ (elderly)"


# ---------------------------------------------------------------------------
# 1. PDF -> text (reuses extraction.py's digital + OCR pipeline)
# ---------------------------------------------------------------------------

def pdf_to_text(pdf_bytes: bytes) -> str:
    """Full text of one PDF, page-joined, digital text with OCR fallback for scans."""
    pages = extraction.extract_pages(pdf_bytes)
    return "\n\n".join(p.text for p in pages)


def split_into_forms(full_text: str) -> List[str]:
    """Split one PDF's full text into one chunk per form.

    Most uploads are one form per PDF, in which case this returns `[full_text]`
    unchanged. For a scanned "booklet" containing several forms back to back,
    each occurrence of a household/respondent-ID line marks where a new form
    starts; the text is cut at each occurrence so every form is extracted (and
    can fail) independently of the others in the same file.
    """
    matches = list(_FORM_BOUNDARY_RE.finditer(full_text))
    if len(matches) < 2:
        return [full_text]
    starts = [m.start() for m in matches]
    starts[0] = 0  # keep any preamble before the first marker with the first form
    return [
        full_text[start : starts[i + 1] if i + 1 < len(starts) else len(full_text)]
        for i, start in enumerate(starts)
    ]


# ---------------------------------------------------------------------------
# 2. LLM field-mapping prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a data-entry engine for household socioeconomic survey forms used "
    "in resettlement/E&S baseline studies. You are given the transcribed text of "
    "ONE completed form. Populate the provided JSON schema using ONLY information "
    "stated on the form. Do not infer, guess, or add facts not present.\n\n"
    "Rules:\n"
    "- Return ONLY valid JSON matching the schema exactly. No prose, no markdown fences.\n"
    "- Use an empty string \"\" for any text field not stated on the form.\n"
    "- Numeric fields (head_age, household_size, rooms, floor_area_sqm, land_area_ha, "
    "income_low, income_high, food_insecure_months): return a plain number, not a string. "
    "Use \"\" only if truly not stated.\n"
    "- income_low / income_high: the form usually states a monthly income RANGE (e.g. "
    "'Approx. USD 85-110') — split it into the two numbers. If only one figure is given, "
    "use it for both income_low and income_high.\n"
    "- food_insecure_months: the number of months per year the form says food is "
    "insufficient. Convert phrasing to a number: 'generally/mostly food secure' -> 0, "
    "'occasional/1-2 months' -> a small number stated, 'chronic .. 6+ months' -> 6. If a "
    "range is given (e.g. '4-5 months') use the higher figure.\n"
    "- vuln_elderly, vuln_female_headed, vuln_disabled, vuln_chronic_illness, "
    "vuln_minority: booleans. true only if that box is ticked/marked with an X or "
    "otherwise indicated as checked on the form's vulnerability indicators line; "
    "false if unmarked.\n"
    "- Canonicalise these fields to EXACTLY one of the listed options (pick the closest "
    "match to what the form states; use \"\" if the form gives no information at all):\n"
    "  tenure_category: one of Owner, Occupant, Tenant\n"
    "  water_source_type: one of Borehole, Unprotected well, River, Unprotected spring, "
    "Piped, Other\n"
    "  sanitation_type: one of Household latrine, Shared latrine, Open defecation, Other\n"
    "  wall_type: one of Mud brick, Concrete block, Other\n"
    "  roof_type: one of Metal, Thatch, Other\n"
    "  electricity_access: Yes or No\n"
    "  education_level (of the household HEAD only): one of None, Primary, Secondary, "
    "Quranic, Other\n"
    "- members: one entry per row in the household members table, IN THE ORDER LISTED "
    "on the form, including the head. age and sex use the same conventions as the member "
    "table (sex as Male/Female from M/F). Capture every row.\n"
    "- project_concern: a one- or two-sentence summary of the form's stated resettlement/"
    "project concerns, in the form's own words as closely as possible (do not add opinion)."
)


def build_prompt(full_text: str) -> Tuple[str, str]:
    """Return (system, user) for the household form population call."""
    import json

    user = (
        f"SCHEMA TO POPULATE (return this shape exactly, with one members[] item per "
        f"household member found):\n{json.dumps(HOUSEHOLD_TEMPLATE, indent=2)}\n\n"
        f"FORM TEXT:\n{full_text}"
    )
    return _SYSTEM, user


def populate_household(full_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Ask the LLM to populate one household record. Returns (record_or_None, raw)."""
    system, user = build_prompt(full_text)
    raw = llm_client.complete_text(system, user, max_tokens=3000)
    parsed = schema_mod._parse_json_lenient(raw)
    if not isinstance(parsed, dict):
        return None, raw
    return parsed, raw


# ---------------------------------------------------------------------------
# 3. Normalisation (type coercion + shape guarantee)
# ---------------------------------------------------------------------------

def _to_number(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None


def _to_str(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "yes", "1", "x")


def normalise_household(data: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee every field exists with the right type. Extraction never invents
    a value — missing/unparseable fields become "" (text) or None (numeric)."""
    out: Dict[str, Any] = {}
    for k in _SCALAR_STR_FIELDS:
        out[k] = _to_str(data.get(k))
    for k in _SCALAR_NUM_FIELDS:
        out[k] = _to_number(data.get(k))
    for k in _VULN_FLAGS:
        out[k] = _to_bool(data.get(k))

    members_raw = data.get("members")
    members: List[Dict[str, Any]] = []
    if isinstance(members_raw, list):
        for m in members_raw:
            if not isinstance(m, dict):
                continue
            row = {
                "name": _to_str(m.get("name")),
                "age": _to_number(m.get("age")),
                "sex": _to_str(m.get("sex")),
                "relationship": _to_str(m.get("relationship")),
                "education": _to_str(m.get("education")),
                "occupation": _to_str(m.get("occupation")),
            }
            if any(v not in ("", None) for v in row.values()):
                members.append(row)
    out["members"] = members
    return out


# ---------------------------------------------------------------------------
# 4. Row builders (household row / analysis row / member rows)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedForm:
    source_file: str
    record: Dict[str, Any]           # normalised, includes "members"
    raw_llm_text: str = ""


def raw_row(form: ExtractedForm) -> Dict[str, Any]:
    """Full row including PII, for the gated raw export. One row per form."""
    row = {"source_file": form.source_file}
    row.update({k: form.record.get(k) for k in _RAW_ORDER if k != "source_file"})
    return row


def analysis_row(form: ExtractedForm) -> Dict[str, Any]:
    """De-identified, analysis-ready row — feeds directly into stats.py."""
    r = form.record
    vuln_count = sum(1 for f in _VULN_FLAGS if r.get(f))
    return {
        "household_id": r.get("household_id") or form.source_file,
        "village": r.get("village"),
        "enumerator": r.get("enumerator"),
        "head_gender": r.get("head_gender"),
        "head_age": r.get("head_age"),
        "head_age_group": age_group(r.get("head_age")),
        "education_level": r.get("education_level"),
        "head_occupation": r.get("head_occupation"),
        "ethnicity": r.get("ethnicity"),
        "religion": r.get("religion"),
        "tenure_category": r.get("tenure_category"),
        "household_size": r.get("household_size"),
        "household_size_band": size_band(r.get("household_size")),
        "rooms": r.get("rooms"),
        "floor_area_sqm": r.get("floor_area_sqm"),
        "wall_type": r.get("wall_type"),
        "roof_type": r.get("roof_type"),
        "water_source_type": r.get("water_source_type"),
        "improved_water": improved_water(r.get("water_source_type")),
        "sanitation_type": r.get("sanitation_type"),
        "electricity_access": r.get("electricity_access"),
        "primary_cooking_fuel": r.get("primary_cooking_fuel"),
        "land_area_ha": r.get("land_area_ha"),
        "livestock": r.get("livestock"),
        "monthly_income_usd": (
            (r["income_low"] + r["income_high"]) / 2
            if r.get("income_low") is not None and r.get("income_high") is not None
            else None
        ),
        "food_insecure_months": r.get("food_insecure_months"),
        "food_security_category": food_security_category(r.get("food_insecure_months")),
        "female_headed": "Yes" if r.get("vuln_female_headed") else "No",
        "vuln_elderly": "Yes" if r.get("vuln_elderly") else "No",
        "vuln_disabled": "Yes" if r.get("vuln_disabled") else "No",
        "vuln_chronic_illness": "Yes" if r.get("vuln_chronic_illness") else "No",
        "vuln_minority": "Yes" if r.get("vuln_minority") else "No",
        "vulnerability_count": vuln_count,
        "vulnerable_household": "Yes" if vuln_count > 0 else "No",
        "project_concern": r.get("project_concern"),
        "source_file": form.source_file,
    }


def member_rows(form: ExtractedForm) -> List[Dict[str, Any]]:
    """De-identified member rows (no names) — one row per household member."""
    hh_id = form.record.get("household_id") or form.source_file
    rows = []
    for m in form.record.get("members", []):
        sex_raw = (m.get("sex") or "").strip().lower()
        sex = "Male" if sex_raw.startswith("m") else ("Female" if sex_raw.startswith("f") else "")
        age = m.get("age")
        rows.append(
            {
                "household_id": hh_id,
                "age": age,
                "sex": sex,
                "relationship": m.get("relationship", ""),
                "age_group": age_group(age),
                "is_dependent": (
                    "Yes" if age is not None and (age < 15 or age >= 65) else
                    ("No" if age is not None else "Unknown")
                ),
                "education": m.get("education", ""),
                "occupation": m.get("occupation", ""),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class FormResult:
    """One form's outcome — success (`form` set) or failure (`error` set)."""
    label: str          # display label, e.g. "hh.pdf" or "booklet.pdf — form 2/5"
    ok: bool
    form: Optional[ExtractedForm] = None
    error: str = ""


def _process_chunk(chunk: str, label: str, source_file: str) -> FormResult:
    """Run one form's text through the size guard, LLM population and the
    quality gate. Never raises — every failure mode becomes a failed FormResult
    so one bad form (oversized, unparseable, not actually a household form)
    cannot abort the rest of a batch or a booklet.
    """
    if len(chunk) > _MAX_CHARS_PER_FORM:
        return FormResult(
            label=label, ok=False,
            error=(
                f"Form text is {len(chunk):,} characters, over the "
                f"{_MAX_CHARS_PER_FORM:,}-character safety limit for one form — "
                "skipped rather than risk an oversized or garbled model call."
            ),
        )
    try:
        parsed, raw = populate_household(chunk)
    except llm_client.LLMError as exc:
        return FormResult(label=label, ok=False, error=f"LLM error: {exc}")
    if parsed is None:
        return FormResult(
            label=label, ok=False,
            error=f"Model output was not valid JSON: {raw[:200]}",
        )
    record = normalise_household(parsed)
    if not record.get("household_id") and not record.get("head_name"):
        return FormResult(
            label=label, ok=False,
            error="No household ID or head name found — this does not look "
                  "like a household survey form.",
        )
    return FormResult(
        label=label, ok=True,
        form=ExtractedForm(source_file=source_file, record=record, raw_llm_text=raw),
    )


def extract_forms(pdf_bytes: bytes, source_file: str) -> List[FormResult]:
    """Full pipeline for one uploaded PDF, which may hold ONE form or a
    "booklet" of several forms back to back (see `split_into_forms`).

    Returns one FormResult per form found in the file — always a list, one
    entry even for a single-form PDF, never raises.
    """
    text = pdf_to_text(pdf_bytes)
    if not text.strip():
        return [
            FormResult(
                label=source_file, ok=False,
                error="No extractable text (blank or unreadable scan).",
            )
        ]
    chunks = split_into_forms(text)
    multi = len(chunks) > 1
    return [
        _process_chunk(
            chunk,
            label=f"{source_file} — form {i}/{len(chunks)}" if multi else source_file,
            source_file=source_file,
        )
        for i, chunk in enumerate(chunks, start=1)
    ]
