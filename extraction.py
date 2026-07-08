"""
extraction.py — PDF ingestion for narrative E&S reports.

Responsibilities:
  1. Page-by-page raw text extraction (page numbers preserved for citation).
  2. Section-hierarchy reconstruction (numbered headings -> nested tree).
  3. Table extraction (pdfplumber) + optional LLM structuring.
  4. Embedded figure/map extraction (PyMuPDF) + LLM captioning.

PROTOTYPE NOTE
--------------
Primary path targets DIGITAL-NATIVE PDFs (selectable text): pdfplumber for
text/tables and PyMuPDF for embedded images. As a fallback, pages that yield
little or no selectable text (i.e. scans / image-only pages) are rendered to an
image via PyMuPDF and run through **pytesseract** (local Tesseract OCR).

Tesseract itself is a system binary, NOT a pip package — it must be installed
separately (e.g. the UB-Mannheim build on Windows). If its executable is not on
PATH, set the TESSERACT_CMD env var to its full path. If pytesseract / Tesseract
is unavailable, OCR is skipped gracefully and only the digital text is kept.

Production narrative-mode ingestion is expected to route scanned reports through
AWS Textract for higher-fidelity OCR; pytesseract here is the prototype stand-in.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pdfplumber

import llm_client


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Page:
    number: int          # 1-based page number as printed / ordered
    text: str
    ocr: bool = False    # True if text came from OCR (scanned/image page)


@dataclass
class Heading:
    number: str          # e.g. "6.3.4" ("" for un-numbered headings)
    title: str
    page: int
    level: int           # depth (1 = top level)
    children: List["Heading"] = field(default_factory=list)


@dataclass
class Figure:
    index: int           # 1-based order of appearance
    page: int
    image_bytes: bytes
    media_type: str
    caption: str = ""
    category: str = ""   # photo | map | diagram


@dataclass
class ExtractedDoc:
    pages: List[Page]
    outline: List[Heading]           # top-level headings (tree via .children)
    tables: List[Dict[str, Any]]     # {"page": n, "rows": [[...]], "markdown": str}
    figures: List[Figure]

    def full_text(self) -> str:
        """Full document text with explicit page markers for grounding."""
        return "\n\n".join(f"[Page {p.number}]\n{p.text}" for p in self.pages)

    def outline_text(self) -> str:
        """Flatten the outline to an indented listing for prompt context."""
        lines: List[str] = []

        def walk(nodes: List[Heading]) -> None:
            for h in nodes:
                indent = "  " * (h.level - 1)
                num = f"{h.number} " if h.number else ""
                lines.append(f"{indent}{num}{h.title}  (p.{h.page})")
                walk(h.children)

        walk(self.outline)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Text extraction
# ---------------------------------------------------------------------------

def extract_pages(
    pdf_bytes: bytes, *, ocr: bool = True, ocr_min_chars: int = 20
) -> List[Page]:
    """Return page-by-page text. Page numbers preserved for citation.

    Digital text is read with pdfplumber first. Any page that yields fewer than
    `ocr_min_chars` characters of selectable text is treated as a scan and (when
    `ocr` is on) re-read via Tesseract OCR — see `_ocr_fill_pages`.
    """
    pages: List[Page] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(Page(number=i, text=text))
    if ocr:
        _ocr_fill_pages(pdf_bytes, pages, ocr_min_chars)
    return pages


def _ocr_fill_pages(pdf_bytes: bytes, pages: List[Page], min_chars: int) -> None:
    """OCR any page whose digital text is below `min_chars`, in place.

    Renders each low-text page to a 300-DPI PNG via PyMuPDF and runs pytesseract.
    Missing OCR dependencies (pytesseract / Pillow) or a missing Tesseract binary
    are swallowed: the digital text (possibly empty) is left untouched so the
    prototype never hard-fails on a scan.
    """
    needy = [p for p in pages if len(p.text.strip()) < min_chars]
    if not needy:
        return

    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError:
        return  # OCR stack unavailable — keep digital text as-is.

    # Allow pointing at a Tesseract install that isn't on PATH (common on Windows).
    tess_cmd = os.environ.get("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for p in needy:
            try:
                page = doc.load_page(p.number - 1)
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img).strip()
            except Exception:  # noqa: BLE001 - Tesseract missing / render error
                continue
            if text:
                p.text = text
                p.ocr = True
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 2. Section-hierarchy extraction
# ---------------------------------------------------------------------------

# Matches leading dotted section numbers, e.g. "6", "6.3", "6.3.4" followed by
# a title on the same line. Trailing dot tolerated ("6.3.4.").
_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\.?\s+(\S.*?)\s*$")


def extract_outline(pages: List[Page]) -> List[Heading]:
    """Reconstruct a nested heading tree from numbered section headings.

    Strategy (deterministic, no LLM needed for numbered reports): scan every
    line, keep those that look like a numbered heading and are short enough to
    be a title rather than a sentence. Nesting is derived purely from the depth
    of the dotted number, which is exactly what makes 6.3.4 nest under 6.3
    under 6 regardless of font rendering.

    Un-numbered headings (e.g. a "References" section) are out of scope for this
    prototype's citation needs; the spec allows an LLM fallback for those, but
    the Acorn report uses numbered sections throughout.
    """
    flat: List[Heading] = []
    for page in pages:
        for raw in page.text.splitlines():
            m = _HEADING_RE.match(raw)
            if not m:
                continue
            number, title = m.group(1), m.group(2).strip()

            # Heuristics to avoid catching numbered prose / list items:
            if len(title) > 90:            # real headings are short
                continue
            if title.endswith((".", ";", ",")) and len(title.split()) > 8:
                continue                    # looks like a sentence
            # A single trailing token that is purely numeric is likely a
            # figure/table caption line ("Table 1 8") — allow, harmless.

            level = number.count(".") + 1
            flat.append(Heading(number=number, title=title, page=page.number, level=level))

    return _build_tree(flat)


def _build_tree(flat: List[Heading]) -> List[Heading]:
    """Turn a flat, in-order heading list into a parent/child tree by level."""
    roots: List[Heading] = []
    stack: List[Heading] = []
    for h in flat:
        while stack and stack[-1].level >= h.level:
            stack.pop()
        if stack:
            stack[-1].children.append(h)
        else:
            roots.append(h)
        stack.append(h)
    return roots


# ---------------------------------------------------------------------------
# 3. Table extraction
# ---------------------------------------------------------------------------

def extract_tables(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract raw tables via pdfplumber. Each retains its page number.

    Raw rows are kept as-is; irregular multi-line cells are cleaned later by
    `structure_table_with_llm()` on demand. We store a markdown rendering too
    so tables can be dropped straight into a prompt as context.
    """
    out: List[Dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            for tbl in page.extract_tables() or []:
                rows = [[(c or "").strip() for c in row] for row in tbl]
                # Drop fully-empty tables (pdfplumber false positives).
                if not any(any(cell for cell in row) for row in rows):
                    continue
                out.append({"page": i, "rows": rows, "markdown": _rows_to_markdown(rows)})
    return out


def _rows_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [r + [""] * (width - len(r)) for r in rows]
    lines = [" | ".join(cell.replace("\n", " ") for cell in r) for r in norm]
    return "\n".join(lines)


def structure_table_with_llm(table: Dict[str, Any]) -> str:
    """Ask Claude to turn one messy table into clean key-value rows (markdown).

    Prompt construction is isolated here so it can be reused by the production
    pipeline. Falls back to the raw markdown if the call fails.
    """
    system = (
        "You clean up tables extracted from a PDF. The raw grid may have merged "
        "or multi-line cells. Return a clean Markdown table with a proper header "
        "row and one record per row. Do not invent data. Return ONLY the table."
    )
    user = f"Raw extracted table (page {table['page']}):\n\n{table['markdown']}"
    try:
        return llm_client.complete_text(system, user, max_tokens=1500)
    except llm_client.LLMError:
        return table["markdown"]


# ---------------------------------------------------------------------------
# 4. Figure / map extraction + captioning
# ---------------------------------------------------------------------------

def extract_figures(pdf_bytes: bytes) -> List[Figure]:
    """Extract embedded raster images via PyMuPDF, in page order, de-duplicated.

    Each embedded image is referenced by an `xref`; the same xref can appear on
    multiple pages (e.g. a logo). We keep the first occurrence of each xref so
    no genuine figure is dropped while repeated assets are not double-counted.
    """
    import fitz  # PyMuPDF; imported lazily so the module imports without it

    figures: List[Figure] = []
    seen_xrefs: set[int] = set()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        counter = 0
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                base = doc.extract_image(xref)
                image_bytes = base["image"]
                ext = base.get("ext", "png")
                # Skip tiny images (icons/rules) that are not real figures.
                if len(image_bytes) < 3000:
                    continue
                counter += 1
                figures.append(
                    Figure(
                        index=counter,
                        page=pno + 1,
                        image_bytes=image_bytes,
                        media_type=f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}",
                    )
                )
    finally:
        doc.close()
    return figures


def caption_figure(fig: Figure) -> Figure:
    """Populate `caption` and `category` for one figure via a vision call.

    Per spec we do NOT interpret map geometry — a one-sentence caption plus a
    category tag (photo / map / diagram) is sufficient. Mutates and returns fig.
    """
    system = (
        "You caption figures from an ecological survey report. Reply in exactly "
        "two lines:\n"
        "CATEGORY: <photo|map|diagram>\n"
        "CAPTION: <one factual sentence describing the image>\n"
        "Do not interpret map coordinates or geometry — describe only what is "
        "visible."
    )
    user = f"This image is figure {fig.index} on page {fig.page} of the report."
    try:
        raw = llm_client.complete_vision(system, user, fig.image_bytes, fig.media_type)
    except llm_client.LLMError as exc:
        fig.category = "unknown"
        fig.caption = f"(captioning failed: {exc})"
        return fig

    for line in raw.splitlines():
        low = line.lower()
        if low.startswith("category:"):
            fig.category = line.split(":", 1)[1].strip().lower()
        elif low.startswith("caption:"):
            fig.caption = line.split(":", 1)[1].strip()
    if not fig.caption:
        fig.caption = raw.strip()
    if not fig.category:
        fig.category = "unknown"
    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_document(pdf_bytes: bytes, *, caption_figures: bool = True) -> ExtractedDoc:
    """Run the full extraction pipeline over a PDF byte string."""
    pages = extract_pages(pdf_bytes)
    outline = extract_outline(pages)
    tables = extract_tables(pdf_bytes)
    figures = extract_figures(pdf_bytes)
    if caption_figures:
        figures = [caption_figure(f) for f in figures]
    return ExtractedDoc(pages=pages, outline=outline, tables=tables, figures=figures)
