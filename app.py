"""
app.py — Narrative Document Analysis (Mode 2, Phase 1) — Streamlit prototype.

Run:
    streamlit run app.py

Auth / config (see llm_client.py):
    AWS credentials  — default AWS credential chain (env vars / ~/.aws profile)
    BEDROCK_MODEL_ID (optional) — model id / ARN; also settable in the sidebar
    BEDROCK_REGION / AWS_REGION (optional) — defaults to us-east-1

SCOPE — this is a standalone PROTOTYPE to validate the extraction schema and
Q&A approach for narrative E&S reports (ESIA chapters, Phase 1 habitat surveys,
biodiversity baselines) BEFORE porting the proven logic into the production
Textract / Bedrock / Pinecone pipeline. It is intentionally minimal:
  - pdfplumber + PyMuPDF for extraction, with pytesseract OCR fallback for scans
  - AWS Bedrock (Amazon Nova Lite by default; any ARN via the sidebar) via the
    Converse API for extraction / captioning / Q&A
  - full-context grounded generation for Q&A (no vector DB / RAG)
See the per-module PROTOTYPE NOTES for what production adds.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

import extraction
import schema as schema_mod
import qa
import report
import llm_client


st.set_page_config(page_title="Narrative Report Analysis", layout="wide")


# ---------------------------------------------------------------------------
# Helpers (defined before use — Streamlit executes this file top-to-bottom)
# ---------------------------------------------------------------------------

def _init_state() -> None:
    st.session_state.setdefault("doc", None)          # extraction.ExtractedDoc
    st.session_state.setdefault("schema", None)       # populated schema dict
    st.session_state.setdefault("schema_raw", "")     # raw LLM text (fallback)
    st.session_state.setdefault("file_name", None)
    st.session_state.setdefault("qa_history", [])     # list[(q, a)]


def _count_headings(nodes) -> int:
    return sum(1 + _count_headings(n.children) for n in nodes)


def _stem() -> str:
    name = st.session_state.file_name or "report"
    return os.path.splitext(name)[0]


def _schema_has_content(schema: dict) -> bool:
    if any(schema.get("site_identity", {}).values()):
        return True
    if schema.get("overall_recommendation"):
        return True
    return any(schema.get(f) for f in schema_mod._ARRAY_FIELDS)


def _render_schema(schema: dict) -> None:
    with st.expander("Site identity", expanded=True):
        st.table(pd.DataFrame(schema["site_identity"].items(), columns=["field", "value"]))

    with st.expander("Overall recommendation", expanded=True):
        st.write(schema.get("overall_recommendation") or "_(not stated)_")

    labels = {
        "habitats_identified": "Habitats identified",
        "species_assessed": "Species assessed",
        "legislation_cited": "Legislation cited",
        "mitigation_measures": "Mitigation measures",
        "invasive_species_flags": "Invasive species flags",
    }
    for field, label in labels.items():
        items = schema.get(field, [])
        with st.expander(f"{label} ({len(items)})", expanded=bool(items)):
            if items:
                st.dataframe(pd.DataFrame(items), use_container_width=True)
            else:
                st.write("_(none found)_")


def _raw_text_export(doc) -> str:
    """Raw transcription with page markers — the raw-export requirement."""
    header = f"# Raw extracted text — {st.session_state.file_name}\n"
    outline = "\n## Section outline\n" + doc.outline_text() + "\n"
    body = "\n## Full text\n" + doc.full_text()
    return header + outline + body


_init_state()


# ---------------------------------------------------------------------------
# Sidebar — config / status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    # Auth is the default AWS credential chain (env vars / ~/.aws profile /
    # instance role) — nothing is pasted into or stored by this app. Only the
    # Bedrock model id and region are set here, held in session memory for this
    # browser session (never written to disk) and re-applied on every rerun.
    model_input = st.text_input(
        "Bedrock model id",
        value=llm_client.active_model(),
        help="Bedrock model id or inference-profile ARN (e.g. an Amazon Nova "
        "Lite ARN). Any Converse-compatible model works. Session-only.",
    )
    region_input = st.text_input("AWS region", value=llm_client.active_region())

    llm_client.set_credentials(model=model_input or None, region=region_input or None)

    creds_ok = llm_client.has_credentials()
    st.write(
        "**AWS credentials:**",
        "✅ default chain" if creds_ok else "❌ none found in default chain",
    )
    st.write("**Region:**", f"`{llm_client.active_region()}`")
    st.write("**Model:**", f"`{llm_client.active_model()}`")
    if not creds_ok:
        st.warning(
            "No AWS credentials found in the default chain. Configure them "
            "(env vars or ~/.aws profile) before running a test."
        )
    st.divider()
    st.caption(
        "Prototype • digital + scanned PDFs (Tesseract OCR fallback) • "
        "AWS Bedrock via Converse (Amazon Nova Lite) • "
        "full-context Q&A (no vector DB)."
    )


st.title("📄 Narrative E&S Report Analysis")
st.caption("Mode 2 (narrative assessment) — Phase 1 prototype")


# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------

uploaded = st.file_uploader("Upload a narrative E&S report (PDF only)", type=["pdf"])

col_a, col_b = st.columns([1, 1])
with col_a:
    run = st.button("🚀 Extract & Analyse", type="primary", disabled=uploaded is None)
with col_b:
    do_captions = st.checkbox("Caption figures (vision calls)", value=True)


if run and uploaded is not None:
    pdf_bytes = uploaded.getvalue()
    st.session_state.file_name = uploaded.name
    st.session_state.qa_history = []

    try:
        with st.status("Extracting document…", expanded=True) as status:
            st.write("Reading text, tables and figures…")
            doc = extraction.extract_document(pdf_bytes, caption_figures=do_captions)
            st.session_state.doc = doc
            ocr_pages = sum(1 for p in doc.pages if p.ocr)
            st.write(
                f"Pages: {len(doc.pages)} · Headings: "
                f"{_count_headings(doc.outline)} · Tables: {len(doc.tables)} · "
                f"Figures: {len(doc.figures)}"
                + (f" · OCR pages: {ocr_pages}" if ocr_pages else "")
            )

            st.write("Populating extraction schema…")
            populated, raw = schema_mod.populate_schema(
                doc.full_text(),
                doc.outline_text(),
                "\n\n".join(t["markdown"] for t in doc.tables),
            )
            st.session_state.schema = populated
            st.session_state.schema_raw = raw
            status.update(label="Extraction complete ✅", state="complete")
    except llm_client.LLMError as exc:
        st.error(f"LLM error: {exc}")
    except Exception as exc:  # noqa: BLE001 - keep the prototype resilient
        st.error(f"Extraction failed: {exc}")


# ---------------------------------------------------------------------------
# Results (only once a document is loaded)
# ---------------------------------------------------------------------------

doc = st.session_state.doc
if doc is not None:
    st.divider()
    st.subheader(f"Results — {st.session_state.file_name}")

    tab_schema, tab_outline, tab_tables, tab_figs, tab_qa = st.tabs(
        ["📋 Schema", "🗂 Section outline", "📊 Tables", "🖼 Figures", "💬 Q&A"]
    )

    # --- Schema tab ----------------------------------------------------------
    with tab_schema:
        populated = st.session_state.schema
        if populated is None:
            st.info("Schema not populated.")
        elif not _schema_has_content(populated) and st.session_state.schema_raw:
            st.warning("Could not parse a clean JSON schema — showing raw model output.")
            st.code(st.session_state.schema_raw, language="json")
        else:
            _render_schema(populated)
            st.download_button(
                "⬇️ Download schema as CSV",
                data=schema_mod.flatten_to_csv(populated),
                file_name=_stem() + "_schema.csv",
                mime="text/csv",
            )

        # Qualitative analysis report (DOCX) — site identity + overall
        # recommendation + key findings + captioned figures + the session's
        # Q&A transcript. This is the qualitative client deliverable (the CSV
        # above is the structured one). Rebuilt on each rerun so it reflects any
        # Q&A asked so far.
        if populated is not None and _schema_has_content(populated):
            st.download_button(
                "⬇️ Download analysis report (DOCX)",
                data=report.make_report_docx(
                    st.session_state.file_name,
                    populated,
                    doc,
                    st.session_state.qa_history,
                ),
                file_name=_stem() + "_analysis_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        # Raw extracted text export (with section headings preserved).
        st.download_button(
            "⬇️ Download raw extracted text",
            data=_raw_text_export(doc),
            file_name=_stem() + "_raw_text.txt",
            mime="text/plain",
        )

    # --- Outline tab ---------------------------------------------------------
    with tab_outline:
        if doc.outline:
            st.text(doc.outline_text())
        else:
            st.info("No numbered section headings detected.")

    # --- Tables tab ----------------------------------------------------------
    with tab_tables:
        if not doc.tables:
            st.info("No tables extracted.")
        for i, tbl in enumerate(doc.tables, start=1):
            with st.expander(f"Table {i} (page {tbl['page']})", expanded=(i == 1)):
                try:
                    df = pd.DataFrame(tbl["rows"][1:], columns=tbl["rows"][0])
                    st.dataframe(df, use_container_width=True)
                except Exception:  # noqa: BLE001 - irregular grid, show markdown
                    st.text(tbl["markdown"])
                if st.button("✨ Clean this table with Claude", key=f"clean_{i}"):
                    with st.spinner("Structuring…"):
                        st.markdown(extraction.structure_table_with_llm(tbl))

    # --- Figures tab ---------------------------------------------------------
    with tab_figs:
        if not doc.figures:
            st.info("No embedded figures found.")
        cols = st.columns(2)
        for i, fig in enumerate(doc.figures):
            with cols[i % 2]:
                st.image(fig.image_bytes, use_container_width=True)
                tag = f"`{fig.category}`" if fig.category else ""
                st.caption(f"**Fig {fig.index}** (p.{fig.page}) {tag} — {fig.caption}")

    # --- Q&A tab -------------------------------------------------------------
    with tab_qa:
        st.write("Ask a question grounded in this document. Answers cite section numbers.")
        question = st.text_input("Your question", key="qa_input")
        if st.button("Ask", disabled=not question):
            try:
                with st.spinner("Thinking…"):
                    answer = qa.answer_question(
                        question,
                        doc.full_text(),
                        doc.outline_text(),
                        st.session_state.schema or {},
                    )
                st.session_state.qa_history.insert(0, (question, answer))
            except llm_client.LLMError as exc:
                st.error(f"LLM error: {exc}")

        for q, a in st.session_state.qa_history:
            with st.container(border=True):
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
