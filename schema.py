"""
schema.py — the fixed extraction schema, its LLM population, and CSV flattening.

Keeping the schema template, the population prompt, and the flattening logic in
one place means the production Bedrock port only has to re-point the LLM call
(inside `populate_schema`) at the Bedrock client — nothing else changes.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Tuple

import llm_client


# The exact schema requested. `populate_schema` fills a deep copy of this and
# guarantees every key is present (empty string / empty list where unknown).
SCHEMA_TEMPLATE: Dict[str, Any] = {
    "site_identity": {
        "site_name": "",
        "location": "",
        "grid_reference": "",
        "client": "",
        "surveyor": "",
        "surveyor_credentials": "",
        "survey_date": "",
    },
    "habitats_identified": [
        {"habitat_type": "", "description": "", "bap_status": ""}
    ],
    "species_assessed": [
        {
            "species_group": "",
            "legal_protection": "",
            "evidence_or_potential": "",
            "recommended_action": "",
        }
    ],
    "legislation_cited": [
        {"act_or_regulation": "", "schedule_or_section": "", "protects": ""}
    ],
    "mitigation_measures": [{"measure": "", "reason": ""}],
    "invasive_species_flags": [
        {"species": "", "location_on_site": "", "control_obligation": ""}
    ],
    "overall_recommendation": "",
}

# Array fields and the item keys we expect — used for CSV flattening and to
# normalise the LLM output so downstream code can rely on shape.
_ARRAY_FIELDS: Dict[str, List[str]] = {
    "habitats_identified": ["habitat_type", "description", "bap_status"],
    "species_assessed": [
        "species_group",
        "legal_protection",
        "evidence_or_potential",
        "recommended_action",
    ],
    "legislation_cited": ["act_or_regulation", "schedule_or_section", "protects"],
    "mitigation_measures": ["measure", "reason"],
    "invasive_species_flags": ["species", "location_on_site", "control_obligation"],
}


def _schema_as_prompt_block() -> str:
    """Empty template (single blank item per array) for the prompt."""
    empty = copy.deepcopy(SCHEMA_TEMPLATE)
    return json.dumps(empty, indent=2)


def build_population_prompt(full_text: str, outline_text: str, tables_md: str) -> Tuple[str, str]:
    """Return (system, user) for the schema-population call. Isolated for reuse."""
    system = (
        "You are an extraction engine for Environmental & Social (E&S) survey "
        "reports. You are given the full text of one narrative report, its "
        "section outline, and its tables. Populate the provided JSON schema "
        "using ONLY information stated in the document.\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON matching the schema exactly. No prose, no "
        "markdown fences.\n"
        "- Use an empty string \"\" for any scalar not stated, and an empty "
        "array [] for any list with no items.\n"
        "- Do not infer, guess, or add facts that are not in the document.\n"
        "- Capture every distinct species group, legislation item, mitigation "
        "measure, habitat, and invasive-species flag you find — one array item "
        "each."
    )
    user = (
        f"SCHEMA TO POPULATE (return this shape exactly):\n{_schema_as_prompt_block()}\n\n"
        f"SECTION OUTLINE:\n{outline_text}\n\n"
        f"TABLES:\n{tables_md if tables_md.strip() else '(none extracted)'}\n\n"
        f"FULL DOCUMENT TEXT:\n{full_text}"
    )
    return system, user


def populate_schema(full_text: str, outline_text: str, tables_md: str) -> Tuple[Dict[str, Any], str]:
    """Populate the schema from document content.

    Returns (schema_dict, raw_response). If the model output cannot be parsed
    as JSON, schema_dict is the empty template and raw_response holds the text
    so the UI can show a raw-text fallback (per requirement 5).
    """
    system, user = build_population_prompt(full_text, outline_text, tables_md)
    raw = llm_client.complete_text(system, user, max_tokens=6000)

    parsed = _parse_json_lenient(raw)
    if parsed is None:
        return copy.deepcopy(SCHEMA_TEMPLATE), raw

    return _normalise(parsed), raw


def _parse_json_lenient(text: str) -> Dict[str, Any] | None:
    """Parse JSON, tolerating stray markdown fences or surrounding prose."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip ```json fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Grab the outermost { ... } span as a last resort.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee every schema key exists with the right type/shape."""
    out = copy.deepcopy(SCHEMA_TEMPLATE)

    # site_identity — scalar fields
    si = data.get("site_identity") or {}
    if isinstance(si, dict):
        for k in out["site_identity"]:
            v = si.get(k, "")
            out["site_identity"][k] = "" if v is None else str(v)

    # array fields
    for field, keys in _ARRAY_FIELDS.items():
        items = data.get(field)
        cleaned: List[Dict[str, str]] = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                row = {k: ("" if item.get(k) is None else str(item.get(k, ""))) for k in keys}
                if any(row.values()):  # drop fully-empty rows
                    cleaned.append(row)
        out[field] = cleaned

    rec = data.get("overall_recommendation", "")
    out["overall_recommendation"] = "" if rec is None else str(rec)
    return out


# ---------------------------------------------------------------------------
# CSV flattening
# ---------------------------------------------------------------------------

def flatten_to_rows(schema: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flatten the array fields into one row per item for CSV export.

    Each row carries a `section` column naming which schema list it came from,
    plus the item's fields, so a compliance reviewer gets one tidy long table.
    """
    rows: List[Dict[str, str]] = []
    for field, keys in _ARRAY_FIELDS.items():
        for item in schema.get(field, []):
            row = {"section": field}
            row.update({k: item.get(k, "") for k in keys})
            rows.append(row)
    return rows


def flatten_to_csv(schema: Dict[str, Any]) -> str:
    """Return CSV text (one row per species/legislation/mitigation/etc. item).

    A leading block of single-value site_identity + overall_recommendation rows
    is included so the export is self-contained for the reviewer.
    """
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header block: site identity + overall recommendation as key/value rows.
    writer.writerow(["section", "field", "value"])
    for k, v in schema.get("site_identity", {}).items():
        writer.writerow(["site_identity", k, v])
    writer.writerow(["overall_recommendation", "", schema.get("overall_recommendation", "")])
    writer.writerow([])  # blank separator row

    # Long table of list items.
    rows = flatten_to_rows(schema)
    if rows:
        # Union of columns across all row types, 'section' first.
        cols: List[str] = ["section"]
        for r in rows:
            for c in r:
                if c not in cols:
                    cols.append(c)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([r.get(c, "") for c in cols])

    return buf.getvalue()
