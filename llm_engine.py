"""
llm_engine.py
-------------
Central AI / LLM layer for the Resume Builder.

All Groq API calls live here.  Each public function:
  1. Constructs a domain-aware SYSTEM PROMPT that conditions the model's
     persona and output format.
  2. Constructs a USER PROMPT with the concrete task.
  3. Calls Groq, extracts the text content, and returns a typed Python object
     (dict / str) — callers never touch raw API responses.

Prompt-engineering principles used throughout:
  - ROLE ASSIGNMENT  → "You are a Senior Technical Recruiter …" shapes tone.
  - OUTPUT CONTRACT  → Explicit JSON schema in every prompt; "respond ONLY
                        with valid JSON, no preamble, no markdown fences".
  - CHAIN-OF-THOUGHT  → Hidden reasoning ("think step by step") is embedded
                        where quality matters (ATS scoring, bullet rewriting).
  - FEW-SHOT EXAMPLES → Provided inline for bullet-point enhancement so the
                        model understands the STAR-method format expected.
  - TEMPERATURE TUNING→ Low (0.3) for JSON parsing; medium (0.7) for creative
                        bullet writing; 0 for ATS matching (deterministic).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from groq import Groq  # type: ignore
from dotenv import load_dotenv  # type: ignore

from utils import safe_parse_json, truncate_text

# ── Environment ──────────────────────────────────────────────────────────────

load_dotenv()
logger = logging.getLogger(__name__)

# ── Groq client (singleton) ──────────────────────────────────────────────────

_client: Groq | None = None

def _get_client() -> Groq:
    """Lazy-initialise the Groq client once per Python process."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set.  Create a .env file with:\n"
                "  GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx"
            )
        _client = Groq(api_key=api_key)
    return _client


# ── Model config ─────────────────────────────────────────────────────────────

# MODEL = "llama3-70b-8192"          # Best quality-speed balance on Groq
# FALLBACK_MODEL = "mixtral-8x7b-32768"

MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# ── Domain → prompt personality mapping ──────────────────────────────────────

_DOMAIN_CONTEXT: dict[str, str] = {
    "ML / AI": (
        "Focus on ML metrics: model accuracy, dataset sizes, inference latency, "
        "F1 scores, loss curves, and production deployment scale. "
        "Use vocabulary: PyTorch, TensorFlow, Hugging Face, ONNX, MLflow, "
        "feature engineering, A/B testing, model serving."
    ),
    "Cloud": (
        "Emphasise cloud-native metrics: uptime %, cost optimisation ($), "
        "request throughput, latency reduction, infrastructure-as-code coverage. "
        "Vocabulary: AWS/GCP/Azure, Terraform, Kubernetes, Docker, CI/CD, "
        "serverless, SLO/SLA."
    ),
    "Cyber Security": (
        "Highlight risk reduction %, vulnerabilities discovered and patched, "
        "compliance frameworks met (ISO 27001, SOC 2, PCI-DSS). "
        "Vocabulary: penetration testing, SIEM, zero-trust, CVE, OWASP Top 10, "
        "incident response, threat modelling."
    ),
    "UI/UX": (
        "Quantify design impact: conversion rate uplift, task success rate, "
        "NPS improvement, accessibility score (WCAG AA), user-testing sessions. "
        "Vocabulary: Figma, Framer, design systems, user research, A/B tests, "
        "interaction design, information architecture."
    ),
    "Web Dev": (
        "Focus on performance wins: Core Web Vitals, Lighthouse scores, "
        "page-load time, API response time, uptime, user growth. "
        "Vocabulary: React, Node.js, REST/GraphQL, PostgreSQL, Redis, "
        "Docker, GitHub Actions, SEO."
    ),
    "Generic Professional": (
        "Focus on KPIs: revenue impact, cost savings, team size managed, "
        "process efficiency gains %, stakeholder satisfaction. "
        "Use business-oriented language suitable for any industry domain."
    ),
}


def _domain_context(domain: str, sub_domain: str | None) -> str:
    """Return the prompt context string for the active domain."""
    key = sub_domain if sub_domain else domain
    return _DOMAIN_CONTEXT.get(key, _DOMAIN_CONTEXT["Generic Professional"])


# ── Low-level call helper ────────────────────────────────────────────────────

def _call_groq(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> str:
    """
    Send a chat-completion request to Groq and return the raw assistant text.

    Retries once with the fallback model if the primary model throws an error.
    """
    client = _get_client()

    def _request(model: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    try:
        result = _request(MODEL)
        logger.info("Groq call succeeded with model %s.", MODEL)
        return result
    except Exception as primary_exc:
        logger.warning(
            "Primary model %s failed (%s); retrying with %s.",
            MODEL, primary_exc, FALLBACK_MODEL,
        )
        try:
            return _request(FALLBACK_MODEL)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Both Groq models failed.\n"
                f"Primary: {primary_exc}\nFallback: {fallback_exc}"
            ) from fallback_exc


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1 — Parse uploaded resume PDF into JSON schema
# ─────────────────────────────────────────────────────────────────────────────

def parse_resume_to_json(raw_text: str) -> dict[str, Any]:
    """
    Use Groq to extract structured data from free-form resume text.

    Prompt design:
      - The system prompt establishes the model as an expert HR data extractor.
      - The output schema is described field-by-field in the system prompt so
        the model understands every key it must populate.
      - Temperature 0.2 → near-deterministic extraction (no creativity needed).

    Parameters
    ----------
    raw_text : str
        Plain text extracted from the uploaded PDF.

    Returns
    -------
    dict
        A dictionary matching (a subset of) get_default_resume_data() keys.
    """
    safe_text = truncate_text(raw_text, max_chars=5000)

    system_prompt = """
You are an expert HR data extraction engine specialised in Indian resume formats.
Your ONLY task is to read the provided resume text and extract information into a
strict JSON schema.

OUTPUT RULES (MANDATORY):
- Respond ONLY with a single valid JSON object.
- Do NOT include any explanatory text, preamble, or markdown code fences.
- If a field is missing from the resume, use null for strings and [] for arrays.
- For education, always try to identify 10th, 12th/Diploma, UG, PG entries.

JSON SCHEMA YOU MUST RETURN:
{
  "full_name": "string",
  "email": "string",
  "phone": "string",
  "city": "string",
  "state": "string",
  "linkedin": "string",
  "github": "string",
  "dob": "DD/MM/YYYY or null",
  "gender": "Male|Female|Other|null",
  "languages_known": ["string"],
  "summary": "string — the professional summary/objective paragraph",
  "education": [
    {
      "level": "10th|12th|Diploma|UG|PG|Other",
      "institution": "string",
      "board_university": "string",
      "year_of_passing": "YYYY",
      "percentage_cgpa": "string",
      "stream": "string"
    }
  ],
  "experience": [
    {
      "company": "string",
      "role": "string",
      "location": "string",
      "start_date": "MM/YYYY",
      "end_date": "MM/YYYY or Present",
      "description": "string"
    }
  ],
  "projects": [
    {
      "title": "string",
      "tech_stack": "string",
      "github_link": "string or null",
      "live_link": "string or null",
      "description": "string"
    }
  ],
  "skills_technical": ["string"],
  "skills_soft": ["string"],
  "skills_tools": ["string"],
  "certifications": [
    {
      "name": "string",
      "issuer": "string",
      "year": "string",
      "url": "string or null"
    }
  ],
  "cocurricular": ["string"]
}
""".strip()

    user_prompt = f"""
Extract all resume data from the following text and return it in the specified JSON schema.

RESUME TEXT:
---
{safe_text}
---
""".strip()

    raw_response = _call_groq(system_prompt, user_prompt, temperature=0.2, max_tokens=3000)
    return safe_parse_json(raw_response)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2 — Enhance bullet points (STAR method)
# ─────────────────────────────────────────────────────────────────────────────

def enhance_bullet_points(
    raw_description: str,
    role_context: str,
    target_jd: str,
    domain: str,
    sub_domain: str | None,
) -> list[str]:
    """
    Rewrite a rough experience / project description into 3 powerful,
    quantified STAR-method bullet points tailored to the target JD.

    Prompt design:
      - FEW-SHOT example baked into the system prompt so the model understands
        the expected style (action verb + task + result + metric).
      - Domain context injected to steer vocabulary (ML metrics vs KPIs).
      - Temperature 0.7 → slightly creative to produce vivid language.
      - Strict JSON array output for easy splitting.

    Returns
    -------
    list[str]
        Exactly 3 bullet point strings (without leading "•" — the template adds that).
    """
    domain_ctx = _domain_context(domain, sub_domain)

    system_prompt = f"""
You are a Senior Technical Resume Writer who has helped 10,000+ Indian professionals
land jobs at top MNCs and startups.  You specialise in the STAR method (Situation,
Task, Action, Result) and quantified achievements.

DOMAIN FOCUS: {domain_ctx}

RULES:
1. Rewrite the candidate's description into EXACTLY 3 bullet points.
2. Each bullet must start with a strong ACTION VERB (past tense for past roles,
   present tense for current roles).
3. Every bullet must include a QUANTIFIED RESULT (%, ₹, x faster, N users, etc.).
   If the original lacks metrics, make a reasonable, defensible estimate marked
   with "~" (e.g., "~30% reduction").
4. Align vocabulary to the Target Job Description keywords where possible.
5. Respond ONLY with a JSON array of 3 strings.  No preamble, no markdown fences.

GOOD EXAMPLE OUTPUT:
[
  "Architected a real-time recommendation engine using collaborative filtering, improving click-through rate by 23% and driving ~₹15L incremental revenue per quarter.",
  "Reduced ML model training time by 40% by migrating pipeline from scikit-learn to PyTorch with mixed-precision training on 4 × A100 GPUs.",
  "Led a 5-member cross-functional team to ship 3 new product features in 6 weeks, achieving a 4.7/5 satisfaction score in post-launch user surveys."
]
""".strip()

    user_prompt = f"""
ROLE / CONTEXT: {role_context}

TARGET JOB DESCRIPTION (use these keywords):
{truncate_text(target_jd, 1500)}

CANDIDATE'S ROUGH DESCRIPTION:
{raw_description}

Rewrite into 3 STAR-method bullet points. Return a JSON array of 3 strings only.
""".strip()

    raw_response = _call_groq(system_prompt, user_prompt, temperature=0.7, max_tokens=800)
    parsed = safe_parse_json(raw_response)

    # The model should return a list directly; handle both list and {"bullets": [...]}
    if isinstance(parsed, list):
        bullets = parsed
    elif isinstance(parsed, dict):
        # Try common wrapper keys
        for key in ("bullets", "points", "bullet_points", "result"):
            if key in parsed and isinstance(parsed[key], list):
                bullets = parsed[key]
                break
        else:
            bullets = list(parsed.values())
    else:
        bullets = [str(parsed)]

    # Guarantee exactly 3 bullets
    bullets = [str(b).strip() for b in bullets if str(b).strip()][:3]
    while len(bullets) < 3:
        bullets.append("(Additional quantified bullet — please fill in manually.)")

    return bullets


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3 — Skill Gap Analysis from JD
# ─────────────────────────────────────────────────────────────────────────────

def analyse_skill_gaps(
    current_skills: list[str],
    target_jd: str,
    domain: str,
    sub_domain: str | None,
) -> dict[str, Any]:
    """
    Compare the candidate's current skills against the target JD and return:
      - A list of JD-required skills the candidate ALREADY has.
      - A list of MISSING skills the candidate should add.
      - A short recommendation paragraph.

    Returns
    -------
    dict with keys: "matched", "missing", "recommendation"
    """
    domain_ctx = _domain_context(domain, sub_domain)

    system_prompt = f"""
You are a Technical Talent Acquisition Specialist with deep expertise in Indian
IT hiring for {domain} — {sub_domain or 'General'}.

DOMAIN CONTEXT: {domain_ctx}

OUTPUT RULES:
- Respond ONLY with a valid JSON object.
- No markdown fences, no preamble.

JSON SCHEMA:
{{
  "matched": ["skill already possessed that JD requires"],
  "missing": ["skill JD requires but candidate lacks"],
  "recommendation": "2-3 sentence actionable advice on which missing skills to prioritise and how."
}}
""".strip()

    skills_str = ", ".join(current_skills) if current_skills else "None listed yet"

    user_prompt = f"""
CANDIDATE'S CURRENT SKILLS:
{skills_str}

TARGET JOB DESCRIPTION:
{truncate_text(target_jd, 2000)}

Analyse the skill gap and return the JSON object.
""".strip()

    raw_response = _call_groq(system_prompt, user_prompt, temperature=0.3, max_tokens=1000)
    result = safe_parse_json(raw_response)

    # Ensure expected keys exist
    return {
        "matched": result.get("matched", []),
        "missing": result.get("missing", []),
        "recommendation": result.get("recommendation", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 4 — ATS Match Score
# ─────────────────────────────────────────────────────────────────────────────

def calculate_ats_score(
    resume_text: str,
    target_jd: str,
    domain: str,
    sub_domain: str | None,
) -> dict[str, Any]:
    """
    Calculate an ATS (Applicant Tracking System) compatibility score between
    the generated resume and the target JD.

    The model acts as an ATS system and returns:
      - score: int (0–100)
      - matched_keywords: list[str]
      - missing_keywords: list[str]
      - section_scores: dict  (breakdown by section)
      - suggestions: str  (top 3 actionable improvements)

    Prompt design:
      - Temperature 0 → fully deterministic scoring.
      - Structured chain-of-thought hidden in the system prompt guides the
        model to score each section before arriving at a total.
    """
    domain_ctx = _domain_context(domain, sub_domain)

    system_prompt = f"""
You are an enterprise ATS (Applicant Tracking System) engine used by Fortune 500
companies and top Indian conglomerates.  You score resumes against job descriptions
with precision and objectivity.

DOMAIN: {domain} — {sub_domain or 'General'}
{domain_ctx}

SCORING METHODOLOGY (think step-by-step internally before outputting):
1. Extract required hard skills, soft skills, tools, and certifications from the JD.
2. Check each against the resume.
3. Score each section: Skills (30 pts), Experience relevance (30 pts),
   Education fit (15 pts), Keyword density (15 pts), Formatting/structure (10 pts).
4. Sum to get total score out of 100.

OUTPUT RULES:
- Respond ONLY with a valid JSON object.
- No markdown fences, no preamble, no commentary outside the JSON.

JSON SCHEMA:
{{
  "score": 0-100,
  "matched_keywords": ["keyword found in both JD and resume"],
  "missing_keywords": ["keyword in JD but NOT in resume"],
  "section_scores": {{
    "skills": 0-30,
    "experience_relevance": 0-30,
    "education_fit": 0-15,
    "keyword_density": 0-15,
    "formatting": 0-10
  }},
  "suggestions": "Top 3 specific, actionable improvements to increase the score."
}}
""".strip()

    user_prompt = f"""
TARGET JOB DESCRIPTION:
{truncate_text(target_jd, 2000)}

CANDIDATE'S RESUME TEXT:
{truncate_text(resume_text, 3000)}

Calculate the ATS score and return the JSON object.
""".strip()

    raw_response = _call_groq(system_prompt, user_prompt, temperature=0.0, max_tokens=1200)
    result = safe_parse_json(raw_response)

    # Normalise score to int
    try:
        result["score"] = int(result.get("score", 0))
    except (TypeError, ValueError):
        result["score"] = 0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 5 — Generate Professional Summary
# ─────────────────────────────────────────────────────────────────────────────

def generate_summary(
    resume_data: dict[str, Any],
    target_jd: str,
    domain: str,
    sub_domain: str | None,
) -> str:
    """
    Auto-generate a 3–4 sentence professional summary tailored to the target JD.

    Returns
    -------
    str
        The summary paragraph (plain text, no bullets).
    """
    domain_ctx = _domain_context(domain, sub_domain)

    # Build a mini-bio from session data for the prompt
    name = resume_data.get("full_name", "The candidate")
    skills = ", ".join(
        (resume_data.get("skills_technical", []) or [])[:8]
    )
    exp_roles = [
        f"{e.get('role','')} at {e.get('company','')}"
        for e in (resume_data.get("experience", []) or [])
    ]
    exp_str = "; ".join(exp_roles[:3]) or "fresher"

    system_prompt = f"""
You are a professional resume copywriter who crafts compelling summaries for
Indian job seekers targeting {domain} roles.

DOMAIN CONTEXT: {domain_ctx}

RULES:
- Write exactly 3–4 sentences.
- Sentence 1: Years of experience, domain, and top specialisation.
- Sentence 2: Key technical strengths / signature achievement.
- Sentence 3: Domain-specific value proposition aligned to the JD.
- Sentence 4 (optional): Soft skill + career aspiration.
- Write in third-person or first-person (be consistent).
- Return ONLY the summary paragraph — no JSON, no bullets, no labels.
""".strip()

    user_prompt = f"""
Candidate Name: {name}
Key Skills: {skills}
Experience: {exp_str}

Target Job Description (extract role title + key requirements):
{truncate_text(target_jd, 1500)}

Write the professional summary now.
""".strip()

    return _call_groq(system_prompt, user_prompt, temperature=0.6, max_tokens=300)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 6 — Generate Interview Questions
# ─────────────────────────────────────────────────────────────────────────────

def generate_interview_questions(
    resume_data: dict[str, Any],
    target_jd: str,
    domain: str,
    sub_domain: str | None,
) -> list[dict]:
    """
    Generate 10 tailored interview questions (4 Technical, 3 HR, 3 Behavioral)
    based on the candidate's resume and the target JD.

    Returns
    -------
    list[dict]
        Each dict: {"question": str, "type": "Technical|HR|Behavioral"}
    """
    domain_ctx = _domain_context(domain, sub_domain)

    name = resume_data.get("full_name", "The candidate")
    skills = ", ".join((resume_data.get("skills_technical", []) or [])[:10])
    exp_roles = [
        f"{e.get('role', '')} at {e.get('company', '')}"
        for e in (resume_data.get("experience", []) or [])
    ]
    exp_str = "; ".join(exp_roles[:3]) or "fresher / no experience listed"
    proj_titles = [
        p.get("title", "") for p in (resume_data.get("projects", []) or [])[:4]
    ]
    proj_str = ", ".join(proj_titles) or "none listed"

    system_prompt = f"""
You are a Senior Technical Interviewer at a top Indian MNC with 15+ years of hiring
experience. You specialise in {domain} — {sub_domain or 'General'}.

DOMAIN CONTEXT: {domain_ctx}

Generate EXACTLY 10 interview questions based on the candidate's resume and the JD.
Breakdown: 4 Technical, 3 HR, 3 Behavioral.

RULES:
- Questions must reference the candidate's ACTUAL skills, projects, and experience.
- Technical questions should probe the gap between the candidate's skills and JD requirements.
- Behavioral questions must use STAR-method prompts ("Tell me about a time when…").
- HR questions cover motivation, salary expectations, notice period, culture fit.
- Respond ONLY with a valid JSON object — no preamble, no markdown fences.

JSON SCHEMA:
{{
  "questions": [
    {{"question": "string", "type": "Technical"}}
  ]
}}
""".strip()

    user_prompt = f"""
CANDIDATE: {name}
KEY SKILLS: {skills}
EXPERIENCE: {exp_str}
PROJECTS: {proj_str}

TARGET JOB DESCRIPTION:
{truncate_text(target_jd, 2000)}

Generate 10 tailored interview questions (4 Technical, 3 HR, 3 Behavioral).
Return the JSON object only.
""".strip()

    raw = _call_groq(system_prompt, user_prompt, temperature=0.5, max_tokens=1500)
    result = safe_parse_json(raw)

    questions = result.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    # Validate each item has the required keys
    clean: list[dict] = []
    for item in questions:
        if isinstance(item, dict) and item.get("question"):
            clean.append({
                "question": str(item.get("question", "")).strip(),
                "type": str(item.get("type", "HR")).strip(),
            })

    logger.info("Generated %d interview questions.", len(clean))
    return clean[:10]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 7 — Evaluate a Candidate's Practice Answer
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_answer(
    question: str,
    candidate_answer: str,
    target_jd: str,
    domain: str,
    sub_domain: str | None,
) -> dict[str, Any]:
    """
    Score a candidate's practice answer out of 10 and return specific feedback.

    Returns
    -------
    dict with keys:
        score          : int (0-10)
        feedback       : str  (what was good + what to improve)
        ideal_answer_hint : str  (outline of a strong answer)
    """
    domain_ctx = _domain_context(domain, sub_domain)

    system_prompt = f"""
You are an expert interview coach for Indian job seekers targeting {domain} roles.

DOMAIN CONTEXT: {domain_ctx}

Evaluate the candidate's answer using these 5 criteria (2 pts each):
1. STAR method usage (Situation → Task → Action → Result)
2. Relevance to the question asked
3. Specificity / quantified results
4. Clarity and logical structure
5. Domain vocabulary alignment with the role

RULES:
- Be honest and constructive — do NOT inflate scores.
- Respond ONLY with a valid JSON object. No preamble, no markdown fences.

JSON SCHEMA:
{{
  "score": 0-10,
  "feedback": "2-3 sentences: what was done well, what is the single most important improvement needed",
  "ideal_answer_hint": "Brief outline (3-4 bullet points as a single string, newline-separated) of what a strong answer would include"
}}
""".strip()

    user_prompt = f"""
INTERVIEW QUESTION:
{question}

CANDIDATE'S ANSWER:
{candidate_answer}

TARGET ROLE CONTEXT (from JD):
{truncate_text(target_jd, 800)}

Evaluate and return the JSON object.
""".strip()

    raw = _call_groq(system_prompt, user_prompt, temperature=0.3, max_tokens=600)
    result = safe_parse_json(raw)

    try:
        result["score"] = max(0, min(10, int(result.get("score", 0))))
    except (TypeError, ValueError):
        result["score"] = 0

    return {
        "score": result.get("score", 0),
        "feedback": result.get("feedback", ""),
        "ideal_answer_hint": result.get("ideal_answer_hint", ""),
    }