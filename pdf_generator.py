"""
pdf_generator.py
----------------
PDF rendering pipeline for the AI Resume Builder.

Workflow:
  1. Receive the full resume data dict (from st.session_state).
  2. Load the appropriate Jinja2 HTML template (indian_modern | ats_friendly).
  3. Render the template with the data → HTML string.
  4. Convert HTML → PDF bytes using xhtml2pdf (primary, pure-Python, works on
     Streamlit Cloud with zero system dependencies) or WeasyPrint (local fallback).
  5. Return:
       - The HTML string  (for in-browser preview via <iframe>).
       - The PDF bytes   (for st.download_button).
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

_jinja_env.filters.setdefault("truncate", lambda s, length=255, killwords=False, end="...", leeway=0:
    s if len(s) <= length else s[:length - len(end)] + end
)

# ── PDF engine detection ──────────────────────────────────────────────────────
# Priority: xhtml2pdf (pure Python, works everywhere including Streamlit Cloud)
#           WeasyPrint (better CSS support but needs system libs — local only)
#           pdfkit     (needs wkhtmltopdf binary)

try:
    from xhtml2pdf import pisa  # type: ignore
    _HAS_XHTML2PDF = True
    logger.info("xhtml2pdf detected — will use as primary PDF engine.")
except ImportError:
    _HAS_XHTML2PDF = False
    logger.warning("xhtml2pdf not found.")

try:
    from weasyprint import HTML as WeasyHTML  # type: ignore
    _HAS_WEASYPRINT = True
    logger.info("WeasyPrint detected — available as secondary PDF engine.")
except Exception:
    _HAS_WEASYPRINT = False

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
    "ats_friendly":  "ats_friendly.html",
}


# ─────────────────────────────────────────────────────────────────────────────
# HTML RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def render_html(resume_data: dict[str, Any], template_name: str = "indian_modern") -> str:
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
    ctx = _sanitise_context(resume_data)
    html = template.render(**ctx)
    logger.info("Template '%s' rendered successfully (%d chars).", template_name, len(html))
    return html


def _sanitise_context(data: dict[str, Any]) -> dict[str, Any]:
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
    Convert an HTML string to PDF bytes.

    Engine priority:
      1. xhtml2pdf  — pure Python, zero system deps, works on Streamlit Cloud.
      2. WeasyPrint — better CSS3 support but needs libpango (local only).
      3. pdfkit     — needs wkhtmltopdf binary.
    """
    if _HAS_XHTML2PDF:
        logger.info("Converting HTML → PDF using xhtml2pdf.")
        try:
            pdf_buffer = io.BytesIO()
            result = pisa.CreatePDF(
                src=io.StringIO(html_string),
                dest=pdf_buffer,
                encoding="utf-8",
            )
            if result.err:
                logger.warning("xhtml2pdf reported errors (err=%s); PDF may still be usable.", result.err)
            pdf_bytes = pdf_buffer.getvalue()
            if pdf_bytes:
                logger.info("xhtml2pdf PDF generated (%d bytes).", len(pdf_bytes))
                return pdf_bytes
            logger.warning("xhtml2pdf returned empty bytes; trying next engine.")
        except Exception as exc:
            logger.warning("xhtml2pdf failed (%s); trying WeasyPrint.", exc)

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
        "Run:  pip install xhtml2pdf"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: Base64-encode PDF for iframe embedding
# ─────────────────────────────────────────────────────────────────────────────

def pdf_bytes_to_b64_uri(pdf_bytes: bytes) -> str:
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT (used by app.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_resume(
    resume_data: dict[str, Any],
    template_name: str = "indian_modern",
) -> tuple[str, bytes]:
    html = render_html(resume_data, template_name)
    pdf  = html_to_pdf_bytes(html)
    return html, pdf


# ─────────────────────────────────────────────────────────────────────────────
# RESUME TEXT SERIALISER (for ATS scoring)
# ─────────────────────────────────────────────────────────────────────────────

def resume_data_to_plain_text(resume_data: dict[str, Any]) -> str:
    lines: list[str] = []

    def add(label: str, value: Any) -> None:
        if value:
            lines.append(f"{label}: {value}")

    add("Name",     resume_data.get("full_name"))
    add("Email",    resume_data.get("email"))
    add("Phone",    resume_data.get("phone"))
    add("Location", f"{resume_data.get('city', '')} {resume_data.get('state', '')}".strip())
    add("LinkedIn", resume_data.get("linkedin"))
    add("GitHub",   resume_data.get("github"))
    add("Summary",  resume_data.get("summary"))

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

    tech  = resume_data.get("skills_technical") or []
    tools = resume_data.get("skills_tools")      or []
    soft  = resume_data.get("skills_soft")       or []
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
