"""
pdf_generator.py
----------------
PDF rendering pipeline for the AI Resume Builder.

PDF Engine: xhtml2pdf (pure Python, zero system dependencies)
Works on: Streamlit Cloud, Windows, Linux, macOS — everywhere.
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
_jinja_env.filters.setdefault(
    "truncate",
    lambda s, length=255, killwords=False, end="...", leeway=0:
        s if len(s) <= length else s[: length - len(end)] + end,
)

# ── PDF engine: xhtml2pdf only (pure Python, no system libs needed) ───────────
try:
    from xhtml2pdf import pisa  # type: ignore
    _HAS_XHTML2PDF = True
    logger.info("xhtml2pdf ready.")
except ImportError:
    _HAS_XHTML2PDF = False
    logger.error("xhtml2pdf not installed. Run: pip install xhtml2pdf")

# ── Template map ──────────────────────────────────────────────────────────────
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
            f"Make sure the 'templates/' folder is next to pdf_generator.py"
        )

    template = _jinja_env.get_template(template_file)
    ctx = _sanitise_context(resume_data)
    html = template.render(**ctx)
    logger.info("Rendered template '%s' (%d chars).", template_name, len(html))
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
# PDF CONVERSION  (xhtml2pdf — pure Python)
# ─────────────────────────────────────────────────────────────────────────────

def html_to_pdf_bytes(html_string: str) -> bytes:
    if not _HAS_XHTML2PDF:
        raise RuntimeError(
            "xhtml2pdf is not installed.\n"
            "Run: pip install xhtml2pdf"
        )

    pdf_buffer = io.BytesIO()
    result = pisa.CreatePDF(
        src=io.StringIO(html_string),
        dest=pdf_buffer,
        encoding="utf-8",
    )
    if result.err:
        logger.warning("xhtml2pdf completed with warnings (err=%s).", result.err)

    pdf_bytes = pdf_buffer.getvalue()
    if not pdf_bytes:
        raise RuntimeError("xhtml2pdf returned empty output. Check the HTML template.")

    logger.info("PDF generated (%d bytes).", len(pdf_bytes))
    return pdf_bytes


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pdf_bytes_to_b64_uri(pdf_bytes: bytes) -> str:
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT  (called by app.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_resume(
    resume_data: dict[str, Any],
    template_name: str = "indian_modern",
) -> tuple[str, bytes]:
    html = render_html(resume_data, template_name)
    pdf  = html_to_pdf_bytes(html)
    return html, pdf


# ─────────────────────────────────────────────────────────────────────────────
# RESUME TEXT SERIALISER  (for ATS scoring)
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
            f"Certification: {cert.get('name')} by "
            f"{cert.get('issuer')} ({cert.get('year')})"
        )

    for item in (resume_data.get("cocurricular") or []):
        lines.append(f"Achievement: {item}")

    return "\n".join(lines)
