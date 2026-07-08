"""
qa.py — grounded natural-language Q&A over a single extracted report.

PROTOTYPE NOTE
--------------
The whole document fits comfortably in the model context, so Q&A is done as
DIRECT grounded generation over the full extracted text + populated schema —
NOT RAG retrieval. Production scale (multi-document knowledge bases, very long
reports) will need Pinecone-based retrieval to select relevant chunks before
generation; that retrieval step would replace the `full_text` argument here.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import llm_client


def build_qa_prompt(
    question: str, full_text: str, outline_text: str, schema: Dict[str, Any]
) -> Tuple[str, str]:
    """Return (system, user) for a grounded Q&A call. Isolated for reuse."""
    system = (
        "You answer questions about a single Environmental & Social survey "
        "report. Answer using ONLY information contained in the provided "
        "document text, outline, and extracted schema.\n\n"
        "Rules:\n"
        "- Ground every claim in the document. Cite the relevant section "
        "number(s) in your answer, e.g. '(Section 4.2.3)'. Use the section "
        "outline to find the right numbers.\n"
        "- If the answer is not stated in the document, reply exactly: "
        "'Not stated in the document.' and briefly say what would be needed. "
        "Never guess or fabricate figures.\n"
        "- Be concise and factual."
    )
    user = (
        f"SECTION OUTLINE:\n{outline_text}\n\n"
        f"EXTRACTED SCHEMA (for quick reference):\n{json.dumps(schema, indent=2)}\n\n"
        f"FULL DOCUMENT TEXT:\n{full_text}\n\n"
        f"QUESTION: {question}"
    )
    return system, user


def answer_question(
    question: str, full_text: str, outline_text: str, schema: Dict[str, Any]
) -> str:
    """Return a grounded answer string with section citations."""
    system, user = build_qa_prompt(question, full_text, outline_text, schema)
    return llm_client.complete_text(system, user, max_tokens=1500)
