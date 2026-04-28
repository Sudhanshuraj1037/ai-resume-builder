"""
pdf_generator.py
----------------
PDF rendering pipeline for the AI Resume Builder.

Workflow:
  1. Receive the full resume data dict (from st.session_state).
  2. Load the appropriate Jinja2 HTML template (indian_modern | ats_friendly).
  3. Render the template with the data → HTML string.
  4. Convert HTML → PDF bytes using WeasyPrint (primary) or pdfkit (fallback).
  5. Return:
       - The HTML string  (for in-browser preview via <iframe>).
       - The PDF bytes   (for st.download_button).

Design notes:
  - Templates live in the `templates/` sub-folder; path is resolved relative
    to THIS file's directory so the app works from any working directory.
  - WeasyPrint is preferred because it handles CSS flexbox, gradients, and
    Base64 image URIs natively — critical for the Indian Modern template.
  - pdfkit (wkhtmltopdf) is the fallback for environments where WeasyPrint's
    Cairo/Pango dependencies are absent (e.g. some Windows setups).
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape  # type: ignore

logger = logging.getLogger(__name__)

# ── Template directory ────────────────────────────────────────────────────────

_TEMPLATE_DIR = Path(__file__).parent / "templates"

# ── Jinja2 Environment ────────────────────────────────────────────────────────

_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
    trim_blocks=True,
    lstrip_blocks=True,
)

# Add the `truncate` filter manually in case autoescape removes it
# (Jinja2's built-in truncate is always available; this is just defensive).
_jinja_env.filters.setdefault("truncate", lambda s, length=255, killwords=False, end="...", leeway=0:
    s if len(s) <= length else s[:length - len(end)] + end
)

# ── WeasyPrint / pdfkit detection ─────────────────────────────────────────────

try:
    from weasyprint import HTML as WeasyHTML  # type: ignore
    _HAS_WEASYPRINT = True
    logger.info("WeasyPrint detected — will use as primary PDF engine.")
except ImportError:
    _HAS_WEASYPRINT = False
    logger.warning("WeasyPrint not found; will attempt pdfkit as fallback.")

try:
    import pdfkit  # type: ignore
    _HAS_PDFKIT = True
    logger.info("pdfkit detected — available as fallback PDF engine.")
except ImportError:
    _HAS_PDFKIT = False


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE NAME → FILE mapping
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_MAP: dict[str, str] = {
    "indian_modern": "indian_modern.html",
    "ats_friendly": "ats_friendly.html",
}


# ─────────────────────────────────────────────────────────────────────────────
# HTML RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def render_html(resume_data: dict[str, Any], template_name: str = "indian_modern") -> str:
    """
    Render a Jinja2 HTML template with the provided resume data.

    Parameters
    ----------
    resume_data : dict
        The full resume data dictionary from st.session_state (see utils.py
        for the canonical schema).
    template_name : str
        One of the keys in TEMPLATE_MAP.

    Returns
    -------
    str
        Rendered HTML string ready for WeasyPrint or browser display.

    Raises
    ------
    KeyError
        If template_name is not in TEMPLATE_MAP.
    FileNotFoundError
        If the template file does not exist on disk.
    """
    template_file = TEMPLATE_MAP.get(template_name)
    if not template_file:
        raise KeyError(
            f"Unknown template '{template_name}'. "
            f"Valid options: {list(TEMPLATE_MAP.keys())}"
        )

    template_path = _TEMPLATE_DIR / template_file
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template file not found: {template_path}\n"
            f"Make sure 'templates/' directory is in the same folder as pdf_generator.py"
        )

    template = _jinja_env.get_template(template_file)

    # Flatten the data dict into keyword arguments for the template.
    # We also coerce None → empty string / empty list for Jinja2's `if` checks.
    ctx = _sanitise_context(resume_data)

    html = template.render(**ctx)
    logger.info(
        "Template '%s' rendered successfully (%d chars).", template_name, len(html)
    )
    return html


def _sanitise_context(data: dict[str, Any]) -> dict[str, Any]:
    """
    Create a Jinja2-safe copy of the resume data:
      - Replace None with "" for string fields.
      - Replace None with [] for list fields.
      - Ensure nested dicts (education, experience, etc.) are lists.
    """
    _LIST_FIELDS = {
        "education", "experience", "projects", "certifications",
        "skills_technical", "skills_soft", "skills_tools",
        "cocurricular", "languages_known",
        "ats_matched_keywords", "ats_missing_keywords",
    }

    ctx: dict[str, Any] = {}
    for key, value in data.items():
        if key in _LIST_FIELDS:
            ctx[key] = value if isinstance(value, list) else []
        elif value is None:
            ctx[key] = ""
        else:
            ctx[key] = value
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# PDF CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def html_to_pdf_bytes(html_string: str) -> bytes:
    """
    Convert an HTML string to PDF bytes using the best available engine.

    Engine priority:
      1. WeasyPrint — handles CSS3, flexbox, gradients, Base64 images.
      2. pdfkit     — requires wkhtmltopdf binary; fallback for WeasyPrint issues.
      3. RuntimeError if neither is available.

    Parameters
    ----------
    html_string : str
        Fully rendered HTML from render_html().

    Returns
    -------
    bytes
        Raw PDF byte content ready for st.download_button or file I/O.
    """
    if _HAS_WEASYPRINT:
        logger.info("Converting HTML → PDF using WeasyPrint.")
        try:
            pdf_bytes = WeasyHTML(string=html_string).write_pdf()
            logger.info("WeasyPrint PDF generated (%d bytes).", len(pdf_bytes))
            return pdf_bytes
        except Exception as exc:
            logger.warning("WeasyPrint failed (%s); trying pdfkit.", exc)

    if _HAS_PDFKIT:
        logger.info("Converting HTML → PDF using pdfkit.")
        try:
            options = {
                "page-size": "A4",
                "margin-top": "0mm",
                "margin-right": "0mm",
                "margin-bottom": "0mm",
                "margin-left": "0mm",
                "encoding": "UTF-8",
                "enable-local-file-access": None,
                "quiet": "",
            }
            pdf_bytes = pdfkit.from_string(html_string, False, options=options)
            logger.info("pdfkit PDF generated (%d bytes).", len(pdf_bytes))
            return pdf_bytes
        except Exception as exc:
            raise RuntimeError(f"pdfkit also failed: {exc}") from exc

    raise RuntimeError(
        "No PDF engine available.\n"
        "Install WeasyPrint:  pip install WeasyPrint\n"
        "  OR\n"
        "Install pdfkit + wkhtmltopdf: pip install pdfkit  (then install wkhtmltopdf binary)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Base64-encode PDF for iframe embedding
# ─────────────────────────────────────────────────────────────────────────────

def pdf_bytes_to_b64_uri(pdf_bytes: bytes) -> str:
    """
    Convert raw PDF bytes to a Base64 data URI suitable for an HTML <iframe>.

    Usage in Streamlit:
        uri = pdf_bytes_to_b64_uri(pdf_bytes)
        st.components.v1.html(
            f'<iframe src="{uri}" width="100%" height="800px"></iframe>',
            height=820,
        )
    """
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT (used by app.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_resume(
    resume_data: dict[str, Any],
    template_name: str = "indian_modern",
) -> tuple[str, bytes]:
    """
    Full pipeline: data → HTML → PDF.

    Parameters
    ----------
    resume_data : dict
        Session state resume data.
    template_name : str
        "indian_modern" or "ats_friendly".

    Returns
    -------
    tuple[str, bytes]
        (html_string, pdf_bytes) — both are returned so the caller can:
          • Show the HTML in a preview iframe.
          • Offer the PDF bytes via st.download_button.
    """
    html = render_html(resume_data, template_name)
    pdf = html_to_pdf_bytes(html)
    return html, pdf


# ─────────────────────────────────────────────────────────────────────────────
# RESUME TEXT SERIALISER (for ATS scoring)
# ─────────────────────────────────────────────────────────────────────────────

def resume_data_to_plain_text(resume_data: dict[str, Any]) -> str:
    """
    Serialise the resume data dict to a flat plain-text string.

    This is passed to llm_engine.calculate_ats_score() so the LLM has a
    clean textual representation of the resume (not HTML tags).
    """
    lines: list[str] = []

    def add(label: str, value: Any) -> None:
        if value:
            lines.append(f"{label}: {value}")

    add("Name", resume_data.get("full_name"))
    add("Email", resume_data.get("email"))
    add("Phone", resume_data.get("phone"))
    add("Location", f"{resume_data.get('city', '')} {resume_data.get('state', '')}".strip())
    add("LinkedIn", resume_data.get("linkedin"))
    add("GitHub", resume_data.get("github"))
    add("Summary", resume_data.get("summary"))

    for edu in (resume_data.get("education") or []):
        lines.append(
            f"Education | {edu.get('level')} | {edu.get('stream')} | "
            f"{edu.get('institution')} | {edu.get('year_of_passing')} | "
            f"{edu.get('percentage_cgpa')}"
        )

    for exp in (resume_data.get("experience") or []):
        lines.append(
            f"Experience | {exp.get('role')} at {exp.get('company')} "
            f"({exp.get('start_date')} - {exp.get('end_date')})"
        )
        if exp.get("description"):
            lines.append(exp["description"])

    for proj in (resume_data.get("projects") or []):
        lines.append(f"Project: {proj.get('title')} | Tech: {proj.get('tech_stack')}")
        if proj.get("description"):
            lines.append(proj["description"])

    tech = resume_data.get("skills_technical") or []
    tools = resume_data.get("skills_tools") or []
    soft = resume_data.get("skills_soft") or []
    all_skills = tech + tools + soft
    if all_skills:
        lines.append("Skills: " + ", ".join(all_skills))

    for cert in (resume_data.get("certifications") or []):
        lines.append(
            f"Certification: {cert.get('name')} by {cert.get('issuer')} ({cert.get('year')})"
        )

    for item in (resume_data.get("cocurricular") or []):
        lines.append(f"Achievement: {item}")

    return "\n".join(lines)