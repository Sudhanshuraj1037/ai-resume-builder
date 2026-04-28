"""
utils.py
--------
Utility / helper module for the AI Resume Builder.

Responsibilities:
  - Extract raw text from uploaded PDF resumes (pdfplumber preferred, PyPDF2 as fallback).
  - Convert uploaded images to Base64 strings so they can be embedded directly
    inside Jinja2 HTML templates (no external file path needed in the rendered PDF).
  - Clean / normalise text coming back from the LLM to strip markdown fences
    before JSON parsing.
  - Provide a safe JSON-parse wrapper that surfaces readable errors.

Design note: Every function is intentionally pure (no Streamlit calls, no global
state mutations).  This keeps them trivially testable in isolation.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

# ── Optional heavy deps (graceful fallback) ─────────────────────────────────

try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    _HAS_PDFPLUMBER = False

try:
    import PyPDF2  # type: ignore
    _HAS_PYPDF2 = True
except ImportError:  # pragma: no cover
    _HAS_PYPDF2 = False

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

# ── Logger ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────


def extract_text_from_pdf(file_obj: Any) -> str:
    """
    Extract plain text from a PDF file object (BytesIO or Streamlit UploadedFile).

    Strategy:
      1. Try pdfplumber (best fidelity for structured Indian resumes — handles
         columns, tables, and multi-font layouts gracefully).
      2. Fall back to PyPDF2 if pdfplumber is unavailable.
      3. Raise RuntimeError if neither library is installed.

    Parameters
    ----------
    file_obj : file-like
        Any file-like object that supports `.read()` and `.seek()`.

    Returns
    -------
    str
        Concatenated text from all pages, pages separated by two newlines.
    """
    # Re-wind to start in case the caller already partially read the buffer.
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    raw_bytes = file_obj.read()
    buffer = io.BytesIO(raw_bytes)

    # ── pdfplumber (preferred) ───────────────────────────────────────────────
    if _HAS_PDFPLUMBER:
        logger.info("Extracting PDF text using pdfplumber.")
        try:
            with pdfplumber.open(buffer) as pdf:
                pages_text = []
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if text:
                        pages_text.append(text.strip())
            extracted = "\n\n".join(pages_text)
            logger.info("pdfplumber extracted %d characters.", len(extracted))
            return extracted
        except Exception as exc:
            logger.warning("pdfplumber failed (%s); trying PyPDF2.", exc)
            buffer.seek(0)

    # ── PyPDF2 (fallback) ────────────────────────────────────────────────────
    if _HAS_PYPDF2:
        logger.info("Extracting PDF text using PyPDF2.")
        try:
            reader = PyPDF2.PdfReader(buffer)
            pages_text = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())
            extracted = "\n\n".join(pages_text)
            logger.info("PyPDF2 extracted %d characters.", len(extracted))
            return extracted
        except Exception as exc:
            logger.error("PyPDF2 also failed: %s", exc)
            raise RuntimeError(f"Could not parse the uploaded PDF: {exc}") from exc

    raise RuntimeError(
        "Neither pdfplumber nor PyPDF2 is installed. "
        "Run: pip install pdfplumber PyPDF2"
    )


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE → BASE64
# ─────────────────────────────────────────────────────────────────────────────


def image_to_base64(file_obj: Any, target_size: tuple[int, int] = (200, 200)) -> str:
    """
    Convert an uploaded image to a Base64-encoded JPEG string so it can be
    embedded directly in the Jinja2 HTML template as a data-URI.

    The function:
      - Resizes the image to *target_size* while maintaining aspect ratio
        (thumbnail mode — never upscales).
      - Converts to RGB so PNG / RGBA photos don't break JPEG encoding.
      - Returns the string WITHOUT the ``data:image/jpeg;base64,`` prefix
        because Jinja2 templates concatenate it themselves (more flexible).

    Parameters
    ----------
    file_obj : file-like
        Streamlit UploadedFile or any BytesIO-compatible object.
    target_size : tuple[int, int]
        (width, height) cap for the thumbnail.  Defaults to 200 × 200 px.

    Returns
    -------
    str
        Pure Base64 string.

    Raises
    ------
    RuntimeError
        If Pillow is not installed or the image cannot be opened.
    """
    if not _HAS_PIL:
        raise RuntimeError(
            "Pillow is required for photo uploads. Run: pip install Pillow"
        )

    if hasattr(file_obj, "seek"):
        file_obj.seek(0)

    try:
        img = Image.open(file_obj)

        # Force to RGB (handles RGBA PNGs, palette images, etc.)
        img = img.convert("RGB")

        # thumbnail() preserves aspect ratio and never upscales.
        img.thumbnail(target_size, Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
        buffer.seek(0)

        b64_str = base64.b64encode(buffer.read()).decode("utf-8")
        logger.info(
            "Image converted to Base64 JPEG (%d chars, size capped at %s).",
            len(b64_str),
            target_size,
        )
        return b64_str

    except Exception as exc:
        logger.error("Image conversion failed: %s", exc)
        raise RuntimeError(f"Could not process the uploaded photo: {exc}") from exc


def file_to_base64(path: str | Path) -> str:
    """
    Read any binary file from disk and return its Base64 representation.
    Useful for embedding local fonts or static assets in the HTML template.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT / JSON HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Regex that strips ```json … ``` or ``` … ``` fences from LLM responses.
_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE | re.MULTILINE
)


def strip_markdown_fences(text: str) -> str:
    """
    Remove Markdown code fences from a string.

    LLMs occasionally wrap their JSON output in triple-backtick blocks even
    when explicitly told not to.  This function extracts the innermost content.

    Examples
    --------
    >>> strip_markdown_fences('```json\\n{"a":1}\\n```')
    '{"a":1}'
    >>> strip_markdown_fences('{"a":1}')
    '{"a":1}'
    """
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _escape_control_chars_in_strings(s: str) -> str:
    """
    Walk through a JSON string character-by-character and escape any literal
    control characters (newline, tab, carriage-return, etc.) that appear
    INSIDE a JSON string value.

    The JSON spec forbids unescaped control characters (U+0000–U+001F) inside
    string literals.  LLMs routinely produce them in long feedback strings,
    causing json.loads to raise 'Invalid control character'.

    This does NOT touch control characters that are structural JSON whitespace
    (outside of string values) — they are left as-is so the parser can still
    use them to tokenise the document.
    """
    result: list[str] = []
    in_string = False
    escape_next = False
    _ESCAPE_MAP = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}

    for ch in s:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ord(ch) < 32:
            # Literal control character inside a string — must be escaped
            result.append(_ESCAPE_MAP.get(ch, f"\\u{ord(ch):04x}"))
        else:
            result.append(ch)

    return "".join(result)


def safe_parse_json(raw: str) -> dict[str, Any]:
    """
    Safely parse a (potentially fence-wrapped) JSON string returned by the LLM.

    Procedure:
      1. Strip Markdown code fences.
      2. Attempt ``json.loads``.
      3. Escape literal control characters inside JSON string values and retry.
         (LLMs often embed raw newlines in long feedback strings, breaking the parser.)
      4. Extract the first ``{`` … last ``}`` block and retry both raw and fixed.
      5. Raise a descriptive ValueError on total failure.

    Parameters
    ----------
    raw : str
        Raw LLM response text.

    Returns
    -------
    dict
        Parsed Python dictionary.

    Raises
    ------
    ValueError
        If JSON cannot be extracted from the response.
    """
    cleaned = strip_markdown_fences(raw)

    # Attempt 1 — direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2 — fix literal control characters inside string values, then retry
    try:
        return json.loads(_escape_control_chars_in_strings(cleaned))
    except json.JSONDecodeError:
        pass

    # Attempt 3 — extract first {...} block (handles LLM preamble text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]

        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

        # Attempt 4 — fix control chars in the extracted snippet and retry
        try:
            return json.loads(_escape_control_chars_in_strings(snippet))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned malformed JSON even after extraction.\n"
                f"Snippet tried: {snippet[:300]}\n"
                f"Error: {exc}"
            ) from exc

    raise ValueError(
        f"No JSON object found in LLM response.\nRaw text (first 500 chars):\n{raw[:500]}"
    )


def clean_text(text: str) -> str:
    """
    Normalise whitespace and remove control characters from a string.
    Useful for cleaning PDF-extracted text before sending to the LLM.
    """
    # Collapse runs of whitespace / newlines into single spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Preserve single newlines; collapse 3+ into double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable control characters (except \n and \t)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 6000) -> str:
    """
    Truncate text to *max_chars* so we don't blow the LLM context window.
    A warning is logged when truncation occurs.
    """
    if len(text) > max_chars:
        logger.warning(
            "Text truncated from %d to %d chars before sending to LLM.",
            len(text),
            max_chars,
        )
        return text[:max_chars] + "\n... [truncated for LLM context]"
    return text


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE SCHEMA  (used by app.py to initialise st.session_state)
# ─────────────────────────────────────────────────────────────────────────────

def get_default_resume_data() -> dict[str, Any]:
    """
    Return the canonical empty resume data dictionary.

    This schema is the single source of truth for ALL fields collected
    across Tabs 1–4 of the Wizard UI.  Centralising it here means:
      - app.py only calls this once on boot.
      - llm_engine.py references the same keys when parsing uploaded resumes.
      - pdf_generator.py Jinja2 templates are guaranteed to have every key.

    Indian Resume–specific fields (DOB, Gender, Languages, Photo, etc.)
    are included as first-class citizens.
    """
    return {
        # ── DOMAIN / META ────────────────────────────────────────────────────
        "domain": None,              # e.g. "Software Engineering"
        "sub_domain": None,          # e.g. "ML / AI", "Cloud", …
        "target_jd": "",             # Paste of the Target Job Description

        # ── PERSONAL INFO (Tab 1) ────────────────────────────────────────────
        "full_name": "",
        "email": "",
        "phone": "",
        "city": "",
        "state": "",
        "address": "",               # Full postal address (optional)
        "linkedin": "",
        "github": "",
        "portfolio": "",             # Personal website / portfolio URL
        "dob": "",                   # DD/MM/YYYY — standard Indian format
        "gender": "",                # Male / Female / Other / Prefer not to say
        "languages_known": [],       # e.g. ["Hindi", "English", "Tamil"]
        "photo_b64": None,           # Base64 JPEG string from utils.image_to_base64

        # ── EDUCATION (Tab 2) ─────────────────────────────────────────────────
        # Each education entry is a dict:
        # {
        #   "level"       : "10th" | "12th" | "Diploma" | "UG" | "PG" | "Other"
        #   "institution" : str
        #   "board_university": str
        #   "year_of_passing": str
        #   "percentage_cgpa": str
        #   "stream"      : str   (Science/PCM, B.Tech CSE, MBA Finance, …)
        # }
        "education": [],

        # ── CO-CURRICULAR / ACHIEVEMENTS (Tab 2) ─────────────────────────────
        "cocurricular": [],          # List of plain strings

        # ── PROFESSIONAL SUMMARY (Tab 3) ─────────────────────────────────────
        "summary": "",

        # ── WORK EXPERIENCE (Tab 3) ──────────────────────────────────────────
        # Each entry is a dict:
        # {
        #   "company"    : str
        #   "role"       : str
        #   "location"   : str
        #   "start_date" : str   (MM/YYYY)
        #   "end_date"   : str   (MM/YYYY or "Present")
        #   "description": str   (raw text OR AI-enhanced bullets)
        # }
        "experience": [],

        # ── PROJECTS (Tab 3) ─────────────────────────────────────────────────
        # Each entry is a dict:
        # {
        #   "title"      : str
        #   "tech_stack" : str   (comma-separated)
        #   "github_link": str   (optional)
        #   "live_link"  : str   (optional)
        #   "description": str   (raw text OR AI-enhanced bullets)
        # }
        "projects": [],

        # ── SKILLS (Tab 4) ───────────────────────────────────────────────────
        "skills_technical": [],      # e.g. ["Python", "React", "Docker"]
        "skills_soft": [],           # e.g. ["Leadership", "Communication"]
        "skills_tools": [],          # e.g. ["Git", "Postman", "Figma"]

        # ── CERTIFICATIONS (Tab 4) ───────────────────────────────────────────
        # Each entry: {"name": str, "issuer": str, "year": str, "url": str}
        "certifications": [],

        # ── ATS ANALYSIS RESULTS (Tab 5) ────────────────────────────────────
        "ats_score": None,           # int 0-100
        "ats_matched_keywords": [],
        "ats_missing_keywords": [],
        "ats_suggestions": "",

        # ── TEMPLATE CHOICE ──────────────────────────────────────────────────
        "selected_template": "indian_modern",   # or "ats_friendly"
    }

# ─────────────────────────────────────────────────────────────────────────────
# GITHUB PROJECT FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_github_projects(github_url: str) -> list[dict]:
    """
    Fetch a user's non-forked public GitHub repositories and return them
    as a list of project dicts matching the resume schema.

    Parameters
    ----------
    github_url : str
        Full GitHub profile URL, e.g. "https://github.com/username"

    Returns
    -------
    list[dict]
        Each dict has keys: title, tech_stack, github_link, live_link, description.

    Raises
    ------
    RuntimeError
        If the GitHub API call fails or the profile is not found.
    """
    import requests

    # Extract username from any URL variant
    username = github_url.rstrip("/").split("/")[-1].lstrip("@")

    logger.info("Fetching GitHub repos for user: %s", username)

    try:
        response = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={"sort": "updated", "per_page": 20},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error while contacting GitHub: {exc}") from exc

    if response.status_code == 404:
        raise RuntimeError(
            f"GitHub profile '{username}' not found. "
            f"Check the URL and make sure the profile is public."
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub API returned status {response.status_code}. Try again in a moment."
        )

    repos = response.json()
    if not isinstance(repos, list):
        raise RuntimeError("Unexpected response from GitHub API.")

    projects: list[dict] = []
    for repo in repos:
        if repo.get("fork", True):          # skip forks — only original work
            continue
        projects.append({
            "title": repo["name"].replace("-", " ").replace("_", " ").title(),
            "tech_stack": repo.get("language", "") or "",
            "github_link": repo.get("html_url", ""),
            "live_link": repo.get("homepage", "") or "",
            "description": repo.get("description", "") or "",
        })

    logger.info("Fetched %d non-forked repos for %s.", len(projects), username)
    return projects