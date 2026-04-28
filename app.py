"""
app.py
------
Main Streamlit entry point for the AI Resume Builder.

Architecture:
  - Uses a TAB-BASED wizard.  All state lives in st.session_state so data
    survives re-renders and tab switches.
  - Sidebar holds global controls (domain, JD input, template selector).
  - Each tab is a self-contained function for readability.
  - Heavy AI operations are called on explicit button press (never on re-render)
    to avoid wasting Groq API credits.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import streamlit as st

# ── Page config MUST be the very first Streamlit call ─────────────────────────
st.set_page_config(
    page_title="AI Resume Builder — India",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal modules ──────────────────────────────────────────────────────────
from utils import get_default_resume_data, image_to_base64, extract_text_from_pdf, fetch_github_projects
from llm_engine import (
    parse_resume_to_json,
    enhance_bullet_points,
    analyse_skill_gaps,
    calculate_ats_score,
    generate_summary,
    generate_interview_questions,
    evaluate_answer,
)
# from pdf_generator import generate_resume, resume_data_to_plain_text, pdf_bytes_to_b64_uri

from pdf_generator import generate_resume, resume_data_to_plain_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (minimal — keeps the Streamlit theme, adds polish)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Section headers in the wizard */
.wizard-section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f3460;
    margin-top: 1.1rem;
    margin-bottom: 0.3rem;
    border-left: 4px solid #e94560;
    padding-left: 0.5rem;
}
/* ATS score gauge */
.ats-score-big {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
}
.ats-score-green { color: #27ae60; }
.ats-score-orange { color: #e67e22; }
.ats-score-red { color: #e74c3c; }
/* Keyword pills */
.kw-matched { background:#d4edda; color:#155724;
              padding:2px 8px; border-radius:12px; margin:2px; display:inline-block; font-size:0.8rem; }
.kw-missing { background:#f8d7da; color:#721c24;
              padding:2px 8px; border-radius:12px; margin:2px; display:inline-block; font-size:0.8rem; }
/* Divider */
.section-divider { border-top: 1px solid #e0e0e0; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    """
    Initialise st.session_state on first load.

    Called at the top of every re-render.  Checks for existence of each key
    before setting it so that user edits are NEVER overwritten by a re-render.
    """
    defaults = get_default_resume_data()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # UI-only flags (not part of the resume data schema)
    if "onboarding_done" not in st.session_state:
        st.session_state["onboarding_done"] = False
    if "preview_html" not in st.session_state:
        st.session_state["preview_html"] = None
    if "preview_pdf_bytes" not in st.session_state:
        st.session_state["preview_pdf_bytes"] = None
    # Interview Prep state
    if "interview_questions" not in st.session_state:
        st.session_state["interview_questions"] = []
    if "interview_feedback" not in st.session_state:
        st.session_state["interview_feedback"] = None


_init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _ss(key: str) -> Any:
    """Shorthand for st.session_state[key]."""
    return st.session_state.get(key)


def _set(key: str, value: Any) -> None:
    """Shorthand for st.session_state[key] = value."""
    st.session_state[key] = value


def _merge_llm_data_into_state(parsed: dict[str, Any]) -> None:
    """
    Merge fields returned by parse_resume_to_json() into session_state.
    Only updates fields that are currently empty to avoid clobbering edits.
    """
    _SCALAR_FIELDS = [
        "full_name", "email", "phone", "city", "state", "address",
        "linkedin", "github", "portfolio", "dob", "gender", "summary",
    ]
    _LIST_FIELDS = [
        "languages_known", "education", "experience", "projects",
        "skills_technical", "skills_soft", "skills_tools",
        "certifications", "cocurricular",
    ]
    for field in _SCALAR_FIELDS:
        if parsed.get(field) and not st.session_state.get(field):
            st.session_state[field] = parsed[field]
    for field in _LIST_FIELDS:
        if parsed.get(field) and not st.session_state.get(field):
            st.session_state[field] = parsed[field]


def _all_skills_flat() -> list[str]:
    """Return a flat list of all skills for skill-gap analysis."""
    return (
        (st.session_state.get("skills_technical") or []) +
        (st.session_state.get("skills_tools") or []) +
        (st.session_state.get("skills_soft") or [])
    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/resume.png", width=64)
        st.title("AI Resume Builder")
        st.caption("Built for the Indian Job Market 🇮🇳")
        st.divider()

        # ── STEP 1: Domain Setup ──────────────────────────────────────────────
        st.markdown("### 🎯 Step 1 — Target Domain")
        domain_options = ["Software Engineering", "Generic Professional"]
        selected_domain = st.selectbox(
            "Select your domain",
            options=domain_options,
            index=domain_options.index(_ss("domain")) if _ss("domain") in domain_options else 0,
            key="domain_selector",
        )
        _set("domain", selected_domain)

        sub_domain = None
        if selected_domain == "Software Engineering":
            sub_options = ["ML / AI", "Cloud", "Cyber Security", "UI/UX", "Web Dev"]
            sub_domain = st.selectbox(
                "Specialisation",
                options=sub_options,
                index=sub_options.index(_ss("sub_domain")) if _ss("sub_domain") in sub_options else 0,
                key="sub_domain_selector",
            )
            _set("sub_domain", sub_domain)
        else:
            _set("sub_domain", None)

        st.divider()

        # ── Target JD input ───────────────────────────────────────────────────
        st.markdown("### 📋 Target Job Description")
        st.caption("Paste the full JD — used for AI enhancements & ATS scoring.")
        jd_text = st.text_area(
            "Job Description",
            value=_ss("target_jd") or "",
            height=220,
            placeholder="Paste the full job description here…",
            label_visibility="collapsed",
        )
        _set("target_jd", jd_text)

        st.divider()

        # ── Template selector ─────────────────────────────────────────────────
        st.markdown("### 🎨 Resume Template")
        template_opts = {
            "🇮🇳 Indian Modern (with photo)": "indian_modern",
            "📝 Strict ATS-Friendly": "ats_friendly",
        }
        selected_label = st.radio(
            "Choose template",
            options=list(template_opts.keys()),
            index=0 if _ss("selected_template") == "indian_modern" else 1,
        )
        _set("selected_template", template_opts[selected_label])

        st.divider()
        st.caption("💡 Tip: Fill all tabs before generating the PDF for best results.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ONBOARDING
# ─────────────────────────────────────────────────────────────────────────────

def _render_onboarding() -> None:
    """Show the upload-vs-scratch choice if not yet done."""
    if st.session_state["onboarding_done"]:
        return

    st.header("👋 Welcome to AI Resume Builder")
    st.subheader("How would you like to start?")
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### 📤 Upload Existing Resume")
            st.markdown(
                "Have a PDF?  We'll extract your data automatically "
                "using AI and pre-fill all fields."
            )
            uploaded_file = st.file_uploader(
                "Upload PDF", type=["pdf"], label_visibility="collapsed"
            )
            if uploaded_file and st.button("Parse & Auto-Fill →", type="primary", use_container_width=True):
                with st.spinner("🤖 Extracting text and parsing with AI…"):
                    try:
                        raw_text = extract_text_from_pdf(uploaded_file)
                        parsed = parse_resume_to_json(raw_text)
                        _merge_llm_data_into_state(parsed)
                        st.session_state["onboarding_done"] = True
                        st.success("✅ Resume parsed!  Review and edit your details below.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Parsing failed: {exc}")
                        logger.error(traceback.format_exc())

    with col2:
        with st.container(border=True):
            st.markdown("### ✏️ Start from Scratch")
            st.markdown(
                "No existing resume?  Fill in your details manually "
                "and let AI craft powerful bullet points for you."
            )
            if st.button("Start Fresh →", type="secondary", use_container_width=True):
                st.session_state["onboarding_done"] = True
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Personal Info & Photo
# ─────────────────────────────────────────────────────────────────────────────

def _tab_personal() -> None:
    st.markdown('<div class="wizard-section-header">👤 Personal Information</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.session_state["full_name"] = st.text_input(
            "Full Name *", value=_ss("full_name") or "", placeholder="Arjun Kumar Sharma"
        )
        cc1, cc2 = st.columns(2)
        with cc1:
            st.session_state["email"] = st.text_input(
                "Email *", value=_ss("email") or "", placeholder="arjun@example.com"
            )
        with cc2:
            st.session_state["phone"] = st.text_input(
                "Phone *", value=_ss("phone") or "", placeholder="+91 98765 43210"
            )

        cc3, cc4 = st.columns(2)
        with cc3:
            st.session_state["city"] = st.text_input(
                "City", value=_ss("city") or "", placeholder="Bengaluru"
            )
        with cc4:
            st.session_state["state"] = st.text_input(
                "State", value=_ss("state") or "", placeholder="Karnataka"
            )

        st.session_state["address"] = st.text_input(
            "Full Address (optional)", value=_ss("address") or "",
            placeholder="Flat 4B, Tech Park Residency, Whitefield, Bengaluru – 560066"
        )

    with c2:
        st.markdown("**Profile Photo**")
        st.caption("Recommended: 300×300 px square, JPG/PNG. Max 2MB.")
        photo_file = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if photo_file:
            with st.spinner("Processing photo…"):
                try:
                    b64 = image_to_base64(photo_file, target_size=(300, 300))
                    st.session_state["photo_b64"] = b64
                    st.image(photo_file, caption="Preview", width=160)
                    st.success("Photo uploaded ✅")
                except Exception as exc:
                    st.error(f"Photo error: {exc}")
        elif _ss("photo_b64"):
            import base64
            st.image(
                base64.b64decode(_ss("photo_b64")),
                caption="Current photo",
                width=160,
            )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wizard-section-header">🔗 Online Profiles</div>', unsafe_allow_html=True)

    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        st.session_state["linkedin"] = st.text_input(
            "LinkedIn URL", value=_ss("linkedin") or "",
            placeholder="linkedin.com/in/arjunkumar"
        )
    with lc2:
        st.session_state["github"] = st.text_input(
            "GitHub URL", value=_ss("github") or "",
            placeholder="github.com/arjunkumar"
        )
    with lc3:
        st.session_state["portfolio"] = st.text_input(
            "Portfolio / Website", value=_ss("portfolio") or "",
            placeholder="arjunkumar.dev"
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wizard-section-header">📋 Additional Details</div>', unsafe_allow_html=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.session_state["dob"] = st.text_input(
            "Date of Birth (DD/MM/YYYY)", value=_ss("dob") or "",
            placeholder="15/08/2000"
        )
        gender_opts = ["", "Male", "Female", "Other", "Prefer not to say"]
        current_gender = _ss("gender") or ""
        gender_idx = gender_opts.index(current_gender) if current_gender in gender_opts else 0
        st.session_state["gender"] = st.selectbox(
            "Gender", options=gender_opts, index=gender_idx
        )
    with dc2:
        lang_raw = st.text_input(
            "Languages Known",
            value=", ".join(_ss("languages_known") or []),
            placeholder="Hindi, English, Tamil, Kannada"
        )
        st.session_state["languages_known"] = [
            l.strip() for l in lang_raw.split(",") if l.strip()
        ]


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Education & Co-Curricular
# ─────────────────────────────────────────────────────────────────────────────

def _tab_education() -> None:
    st.markdown('<div class="wizard-section-header">🎓 Education</div>', unsafe_allow_html=True)
    st.caption(
        "Add entries from 10th to Postgraduate in reverse chronological order. "
        "Indian Boards (CBSE, ICSE, State) and Universities are fully supported."
    )

    education: list[dict] = st.session_state.get("education") or []

    # ── Display existing entries ─────────────────────────────────────────────
    for i, edu in enumerate(education):
        with st.expander(
            f"📘 {edu.get('level','Entry')} — {edu.get('institution', 'Institution')}",
            expanded=(i == 0),
        ):
            ec1, ec2 = st.columns(2)
            level_opts = ["10th", "12th", "Diploma", "UG", "PG", "Other"]
            lvl = edu.get("level", "UG")
            education[i]["level"] = ec1.selectbox(
                "Level", level_opts,
                index=level_opts.index(lvl) if lvl in level_opts else 0,
                key=f"edu_level_{i}",
            )
            education[i]["stream"] = ec2.text_input(
                "Stream / Degree", value=edu.get("stream", ""),
                placeholder="B.Tech Computer Science",
                key=f"edu_stream_{i}",
            )
            ec3, ec4 = st.columns(2)
            education[i]["institution"] = ec3.text_input(
                "Institution", value=edu.get("institution", ""),
                placeholder="IIT Bombay",
                key=f"edu_inst_{i}",
            )
            education[i]["board_university"] = ec4.text_input(
                "Board / University", value=edu.get("board_university", ""),
                placeholder="CBSE / Mumbai University",
                key=f"edu_board_{i}",
            )
            ec5, ec6 = st.columns(2)
            education[i]["year_of_passing"] = ec5.text_input(
                "Year of Passing", value=edu.get("year_of_passing", ""),
                placeholder="2024",
                key=f"edu_year_{i}",
            )
            education[i]["percentage_cgpa"] = ec6.text_input(
                "Percentage / CGPA", value=edu.get("percentage_cgpa", ""),
                placeholder="8.7 CGPA / 87%",
                key=f"edu_cgpa_{i}",
            )
            if st.button("🗑 Remove this entry", key=f"del_edu_{i}"):
                education.pop(i)
                st.session_state["education"] = education
                st.rerun()

    st.session_state["education"] = education

    if st.button("➕ Add Education Entry"):
        education.append({
            "level": "UG", "institution": "", "board_university": "",
            "year_of_passing": "", "percentage_cgpa": "", "stream": "",
        })
        st.session_state["education"] = education
        st.rerun()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="wizard-section-header">🏆 Co-Curricular & Achievements</div>', unsafe_allow_html=True)
    st.caption("Hackathons, sports, cultural events, NSS/NCC, leadership roles, awards…")

    cocurr: list[str] = st.session_state.get("cocurricular") or []
    for i, item in enumerate(cocurr):
        c1, c2 = st.columns([10, 1])
        cocurr[i] = c1.text_input(
            f"Achievement {i+1}", value=item, label_visibility="collapsed",
            key=f"cocurr_{i}",
            placeholder="1st Place — Smart India Hackathon 2023, MHRD"
        )
        if c2.button("✕", key=f"del_cocurr_{i}", help="Remove"):
            cocurr.pop(i)
            st.session_state["cocurricular"] = cocurr
            st.rerun()

    st.session_state["cocurricular"] = cocurr

    if st.button("➕ Add Achievement"):
        cocurr.append("")
        st.session_state["cocurricular"] = cocurr
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Experience & Projects
# ─────────────────────────────────────────────────────────────────────────────

def _tab_experience() -> None:
    if not _ss("target_jd"):
        st.info("💡 Paste your Target Job Description in the sidebar to unlock AI bullet enhancement.")

    # ── Professional Summary ────────────────────────────────────────────────
    st.markdown('<div class="wizard-section-header">📝 Professional Summary</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1:
        st.session_state["summary"] = st.text_area(
            "Summary", value=_ss("summary") or "", height=100,
            placeholder="Results-driven software engineer with 3+ years of experience…",
            label_visibility="collapsed",
        )
    with c2:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("✨ Generate with AI", use_container_width=True):
            if not _ss("target_jd"):
                st.warning("Add a Job Description in the sidebar first.")
            else:
                with st.spinner("Writing summary…"):
                    try:
                        summary = generate_summary(
                            dict(st.session_state),
                            _ss("target_jd"),
                            _ss("domain") or "Generic Professional",
                            _ss("sub_domain"),
                        )
                        st.session_state["summary"] = summary
                        st.rerun()
                    except Exception as exc:
                        st.error(f"AI error: {exc}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Work Experience ─────────────────────────────────────────────────────
    st.markdown('<div class="wizard-section-header">💼 Work Experience</div>', unsafe_allow_html=True)
    experience: list[dict] = st.session_state.get("experience") or []

    for i, exp in enumerate(experience):
        with st.expander(
            f"🏢 {exp.get('role','Role')} @ {exp.get('company','Company')}",
            expanded=(i == 0),
        ):
            ec1, ec2 = st.columns(2)
            experience[i]["role"] = ec1.text_input(
                "Job Title / Role", value=exp.get("role", ""),
                placeholder="Software Engineer II",
                key=f"exp_role_{i}",
            )
            experience[i]["company"] = ec2.text_input(
                "Company", value=exp.get("company", ""),
                placeholder="Infosys Limited",
                key=f"exp_company_{i}",
            )
            ec3, ec4, ec5 = st.columns(3)
            experience[i]["location"] = ec3.text_input(
                "Location", value=exp.get("location", ""),
                placeholder="Bengaluru, KA",
                key=f"exp_loc_{i}",
            )
            experience[i]["start_date"] = ec4.text_input(
                "Start (MM/YYYY)", value=exp.get("start_date", ""),
                placeholder="06/2022",
                key=f"exp_start_{i}",
            )
            experience[i]["end_date"] = ec5.text_input(
                "End (MM/YYYY or 'Present')", value=exp.get("end_date", ""),
                placeholder="Present",
                key=f"exp_end_{i}",
            )

            # Bullet edit + AI enhance
            desc_col, btn_col = st.columns([5, 1])
            with desc_col:
                experience[i]["description"] = st.text_area(
                    "Responsibilities / Achievements",
                    value=exp.get("description", ""),
                    height=120,
                    placeholder=(
                        "• Built REST APIs in Node.js for the payments module\n"
                        "• Reduced DB query time by optimising indexes\n"
                        "• Mentored 2 junior developers"
                    ),
                    key=f"exp_desc_{i}",
                )
            with btn_col:
                st.markdown("<br/><br/><br/>", unsafe_allow_html=True)
                if st.button("✨ Enhance\nvia AI", key=f"enhance_exp_{i}", use_container_width=True):
                    if not _ss("target_jd"):
                        st.warning("Add a JD first.")
                    elif not experience[i]["description"].strip():
                        st.warning("Add some description first.")
                    else:
                        with st.spinner("Rewriting with STAR method…"):
                            try:
                                bullets = enhance_bullet_points(
                                    experience[i]["description"],
                                    f"{experience[i]['role']} at {experience[i]['company']}",
                                    _ss("target_jd"),
                                    _ss("domain") or "Generic Professional",
                                    _ss("sub_domain"),
                                )
                                experience[i]["description"] = "\n".join(f"• {b}" for b in bullets)
                                st.session_state["experience"] = experience
                                st.rerun()
                            except Exception as exc:
                                st.error(f"AI error: {exc}")

            if st.button("🗑 Remove", key=f"del_exp_{i}"):
                experience.pop(i)
                st.session_state["experience"] = experience
                st.rerun()

    st.session_state["experience"] = experience

    if st.button("➕ Add Work Experience"):
        experience.append({
            "company": "", "role": "", "location": "",
            "start_date": "", "end_date": "Present", "description": "",
        })
        st.session_state["experience"] = experience
        st.rerun()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Projects ───────────────────────────────────────────────────────────
    st.markdown('<div class="wizard-section-header">🚀 Projects</div>', unsafe_allow_html=True)

    # ── GitHub Auto-Populate ───────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("**🐙 Auto-fill Projects from GitHub**")
        st.caption("Paste your GitHub profile URL — we'll import all your original (non-forked) repos instantly.")
        gh_col1, gh_col2 = st.columns([4, 1])
        with gh_col1:
            github_input = st.text_input(
                "GitHub URL",
                value=_ss("github") or "",
                placeholder="https://github.com/yourusername",
                label_visibility="collapsed",
                key="github_fetch_input",
            )
        with gh_col2:
            if st.button("⚡ Fetch Repos", use_container_width=True):
                if not github_input.strip():
                    st.warning("Enter a GitHub profile URL first.")
                else:
                    with st.spinner("Fetching your GitHub repositories…"):
                        try:
                            fetched = fetch_github_projects(github_input.strip())
                            if fetched:
                                st.session_state["projects"] = fetched
                                st.success(f"✅ Imported {len(fetched)} repositories! Review and enhance them below.")
                                st.rerun()
                            else:
                                st.warning("No original (non-forked) public repos found on this profile.")
                        except Exception as exc:
                            st.error(f"GitHub fetch failed: {exc}")

    projects: list[dict] = st.session_state.get("projects") or []

    for i, proj in enumerate(projects):
        with st.expander(f"🔨 {proj.get('title','Project')}", expanded=(i == 0)):
            pc1, pc2 = st.columns(2)
            projects[i]["title"] = pc1.text_input(
                "Project Title", value=proj.get("title", ""),
                placeholder="AI-Powered Resume Builder",
                key=f"proj_title_{i}",
            )
            projects[i]["tech_stack"] = pc2.text_input(
                "Tech Stack", value=proj.get("tech_stack", ""),
                placeholder="Python, Streamlit, Groq API, WeasyPrint",
                key=f"proj_tech_{i}",
            )
            pl1, pl2 = st.columns(2)
            projects[i]["github_link"] = pl1.text_input(
                "GitHub Link", value=proj.get("github_link", "") or "",
                placeholder="github.com/user/repo",
                key=f"proj_gh_{i}",
            )
            projects[i]["live_link"] = pl2.text_input(
                "Live Demo URL", value=proj.get("live_link", "") or "",
                placeholder="resumebuilder.streamlit.app",
                key=f"proj_live_{i}",
            )

            desc_col, btn_col = st.columns([5, 1])
            with desc_col:
                projects[i]["description"] = st.text_area(
                    "Project Description",
                    value=proj.get("description", ""),
                    height=120,
                    placeholder=(
                        "Built an AI-powered resume builder using Streamlit and Groq.\n"
                        "Integrated PDF generation, ATS scoring, and STAR method bullets."
                    ),
                    key=f"proj_desc_{i}",
                )
            with btn_col:
                st.markdown("<br/><br/><br/>", unsafe_allow_html=True)
                if st.button("✨ Enhance\nvia AI", key=f"enhance_proj_{i}", use_container_width=True):
                    if not _ss("target_jd"):
                        st.warning("Add a JD first.")
                    elif not projects[i]["description"].strip():
                        st.warning("Add some description first.")
                    else:
                        with st.spinner("Rewriting…"):
                            try:
                                bullets = enhance_bullet_points(
                                    projects[i]["description"],
                                    f"Project: {projects[i]['title']} | Tech: {projects[i]['tech_stack']}",
                                    _ss("target_jd"),
                                    _ss("domain") or "Generic Professional",
                                    _ss("sub_domain"),
                                )
                                projects[i]["description"] = "\n".join(f"• {b}" for b in bullets)
                                st.session_state["projects"] = projects
                                st.rerun()
                            except Exception as exc:
                                st.error(f"AI error: {exc}")

            if st.button("🗑 Remove", key=f"del_proj_{i}"):
                projects.pop(i)
                st.session_state["projects"] = projects
                st.rerun()

    st.session_state["projects"] = projects

    if st.button("➕ Add Project"):
        projects.append({
            "title": "", "tech_stack": "", "github_link": "",
            "live_link": "", "description": "",
        })
        st.session_state["projects"] = projects
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Skills & Certifications
# ─────────────────────────────────────────────────────────────────────────────

def _tab_skills() -> None:
    st.markdown('<div class="wizard-section-header">🛠 Technical Skills</div>', unsafe_allow_html=True)
    st.caption("Enter comma-separated lists.")

    tech_raw = st.text_area(
        "Technical Skills",
        value=", ".join(_ss("skills_technical") or []),
        height=80,
        placeholder="Python, Java, C++, Machine Learning, Deep Learning, SQL, React, FastAPI",
    )
    st.session_state["skills_technical"] = [s.strip() for s in tech_raw.split(",") if s.strip()]

    tools_raw = st.text_area(
        "Tools & Platforms",
        value=", ".join(_ss("skills_tools") or []),
        height=70,
        placeholder="Docker, Kubernetes, AWS, Git, Postman, Jira, Figma",
    )
    st.session_state["skills_tools"] = [s.strip() for s in tools_raw.split(",") if s.strip()]

    soft_raw = st.text_area(
        "Soft Skills",
        value=", ".join(_ss("skills_soft") or []),
        height=60,
        placeholder="Leadership, Communication, Problem Solving, Agile, Teamwork",
    )
    st.session_state["skills_soft"] = [s.strip() for s in soft_raw.split(",") if s.strip()]

    # ── AI Skill Gap Analysis ───────────────────────────────────────────────
    if st.button("🔍 Analyse Skill Gaps from JD", type="primary"):
        if not _ss("target_jd"):
            st.warning("Paste your Target JD in the sidebar first.")
        else:
            with st.spinner("Comparing your skills to the JD…"):
                try:
                    result = analyse_skill_gaps(
                        _all_skills_flat(),
                        _ss("target_jd"),
                        _ss("domain") or "Generic Professional",
                        _ss("sub_domain"),
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"✅ Matched ({len(result['matched'])})")
                        st.markdown(
                            " ".join(f'<span class="kw-matched">{k}</span>' for k in result["matched"]),
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.error(f"❌ Missing ({len(result['missing'])})")
                        st.markdown(
                            " ".join(f'<span class="kw-missing">{k}</span>' for k in result["missing"]),
                            unsafe_allow_html=True,
                        )
                    if result["recommendation"]:
                        st.info(f"💡 {result['recommendation']}")
                except Exception as exc:
                    st.error(f"AI error: {exc}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Certifications ──────────────────────────────────────────────────────
    st.markdown('<div class="wizard-section-header">📜 Certifications</div>', unsafe_allow_html=True)
    certifications: list[dict] = st.session_state.get("certifications") or []

    for i, cert in enumerate(certifications):
        with st.expander(f"🏅 {cert.get('name', 'Certification')}", expanded=False):
            cc1, cc2 = st.columns(2)
            certifications[i]["name"] = cc1.text_input(
                "Certification Name", value=cert.get("name", ""),
                placeholder="AWS Certified Solutions Architect",
                key=f"cert_name_{i}",
            )
            certifications[i]["issuer"] = cc2.text_input(
                "Issuing Organisation", value=cert.get("issuer", ""),
                placeholder="Amazon Web Services",
                key=f"cert_issuer_{i}",
            )
            cc3, cc4 = st.columns(2)
            certifications[i]["year"] = cc3.text_input(
                "Year", value=cert.get("year", ""),
                placeholder="2024",
                key=f"cert_year_{i}",
            )
            certifications[i]["url"] = cc4.text_input(
                "Credential URL", value=cert.get("url", "") or "",
                placeholder="credly.com/badges/…",
                key=f"cert_url_{i}",
            )
            if st.button("🗑 Remove", key=f"del_cert_{i}"):
                certifications.pop(i)
                st.session_state["certifications"] = certifications
                st.rerun()

    st.session_state["certifications"] = certifications
    if st.button("➕ Add Certification"):
        certifications.append({"name": "", "issuer": "", "year": "", "url": ""})
        st.session_state["certifications"] = certifications
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — ATS Matcher
# ─────────────────────────────────────────────────────────────────────────────

def _tab_ats() -> None:
    st.markdown('<div class="wizard-section-header">🤖 ATS Match Score</div>', unsafe_allow_html=True)
    st.markdown(
        "Our AI acts as an enterprise ATS engine to score your resume against "
        "the Target Job Description.  Get a score, see matched/missing keywords, "
        "and receive improvement suggestions."
    )

    if not _ss("target_jd"):
        st.warning("⚠️ Please paste your Target Job Description in the sidebar first.")
        return

    if st.button("⚡ Calculate ATS Score Now", type="primary", use_container_width=True):
        with st.spinner("AI is analysing your resume…  (may take 10–20 seconds)"):
            try:
                resume_text = resume_data_to_plain_text(dict(st.session_state))
                result = calculate_ats_score(
                    resume_text,
                    _ss("target_jd"),
                    _ss("domain") or "Generic Professional",
                    _ss("sub_domain"),
                )
                # Save to session state so it persists across re-renders
                st.session_state["ats_score"] = result.get("score", 0)
                st.session_state["ats_matched_keywords"] = result.get("matched_keywords", [])
                st.session_state["ats_missing_keywords"] = result.get("missing_keywords", [])
                st.session_state["ats_suggestions"] = result.get("suggestions", "")
                st.session_state["ats_section_scores"] = result.get("section_scores", {})
            except Exception as exc:
                st.error(f"ATS scoring failed: {exc}")
                logger.error(traceback.format_exc())

    # Display results if available
    score = _ss("ats_score")
    if score is not None:
        st.divider()

        # Big score number
        colour_class = (
            "ats-score-green" if score >= 70
            else "ats-score-orange" if score >= 45
            else "ats-score-red"
        )
        st.markdown(
            f'<div class="ats-score-big {colour_class}">{score} / 100</div>',
            unsafe_allow_html=True,
        )
        st.progress(score / 100)

        interpretation = (
            "🟢 Strong match!  You're well-positioned for this role." if score >= 70
            else "🟡 Moderate match.  Address missing keywords to improve." if score >= 45
            else "🔴 Low match.  Significant gaps need attention before applying."
        )
        st.info(interpretation)

        # Section breakdown
        section_scores = st.session_state.get("ats_section_scores", {})
        if section_scores:
            st.subheader("Section Breakdown")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            labels = {
                "skills": ("Skills", 30, sc1),
                "experience_relevance": ("Experience", 30, sc2),
                "education_fit": ("Education", 15, sc3),
                "keyword_density": ("Keywords", 15, sc4),
                "formatting": ("Formatting", 10, sc5),
            }
            for key, (label, max_pts, col) in labels.items():
                val = section_scores.get(key, 0)
                col.metric(label, f"{val}/{max_pts}")

        # Keywords
        matched = _ss("ats_matched_keywords") or []
        missing = _ss("ats_missing_keywords") or []

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"✅ Matched Keywords ({len(matched)})")
            st.markdown(
                " ".join(f'<span class="kw-matched">{k}</span>' for k in matched) or "—",
                unsafe_allow_html=True,
            )
        with col2:
            st.subheader(f"❌ Missing Keywords ({len(missing)})")
            st.markdown(
                " ".join(f'<span class="kw-missing">{k}</span>' for k in missing) or "—",
                unsafe_allow_html=True,
            )

        # Suggestions
        suggestions = _ss("ats_suggestions")
        if suggestions:
            st.subheader("💡 Top Suggestions")
            st.markdown(suggestions)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Preview & Download
# ─────────────────────────────────────────────────────────────────────────────

def _tab_preview() -> None:
    st.markdown('<div class="wizard-section-header">👁 Preview & Download</div>', unsafe_allow_html=True)
    st.markdown(
        f"**Selected template:** `{_ss('selected_template')}`  "
        f"(Change in the sidebar)"
    )

    st.info(
        "Click **Preview Resume** to render your resume.  "
        "The PDF preview appears below.  "
        "Then click **Download as PDF** to save it."
    )

    preview_col, _ = st.columns([1, 2])
    with preview_col:
        if st.button("🖥 Preview Resume", type="primary", use_container_width=True):
            with st.spinner("Rendering your resume…"):
                try:
                    html, pdf_bytes = generate_resume(
                        dict(st.session_state),
                        _ss("selected_template") or "indian_modern",
                    )
                    st.session_state["preview_html"] = html
                    st.session_state["preview_pdf_bytes"] = pdf_bytes
                    st.success("Resume rendered successfully!")
                except Exception as exc:
                    st.error(f"Rendering failed: {exc}\n\n{traceback.format_exc()}")

    # ── Show PDF iframe ───────────────────────────────────────────────────
    # pdf_bytes = st.session_state.get("preview_pdf_bytes")
    # if pdf_bytes:
    #     st.divider()
    #     st.subheader("📄 Resume Preview")
    #     pdf_uri = pdf_bytes_to_b64_uri(pdf_bytes)
    #     st.components.v1.html(
    #         f'<iframe src="{pdf_uri}" width="100%" height="900" '
    #         f'style="border:1px solid #ddd; border-radius:6px;"></iframe>',
    #         height=920,
    #     )
    
    pdf_bytes = st.session_state.get("preview_pdf_bytes")
    html_str  = st.session_state.get("preview_html", "")
    if pdf_bytes:
        st.divider()
        st.subheader("📄 Resume Preview")
        # Show HTML preview (Chrome blocks base64 PDF iframes)
        st.components.v1.html(html_str, height=950, scrolling=True)
        
        

        st.divider()
        candidate_name = (_ss("full_name") or "resume").replace(" ", "_")
        st.download_button(
            label="⬇ Download Resume as PDF",
            data=pdf_bytes,
            file_name=f"{candidate_name}_Resume.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )

        # Also offer HTML download (useful for debugging)
        html_str = st.session_state.get("preview_html", "")
        if html_str:
            with st.expander("🛠 Developer: Download raw HTML"):
                st.download_button(
                    "⬇ Download HTML",
                    data=html_str,
                    file_name=f"{candidate_name}_Resume.html",
                    mime="text/html",
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — Interview Prep
# ─────────────────────────────────────────────────────────────────────────────

def _tab_interview_prep() -> None:
    st.markdown('<div class="wizard-section-header">🎯 Interview Preparation</div>', unsafe_allow_html=True)
    st.markdown(
        "AI studies your resume + target JD to generate personalised interview questions. "
        "Type your practice answer and get an instant score with coaching feedback."
    )

    if not _ss("target_jd"):
        st.warning("⚠️ Paste your Target Job Description in the sidebar to generate tailored questions.")
        return

    # ── Generate Questions ──────────────────────────────────────────────────
    if st.button("🎲 Generate My Interview Questions", type="primary", use_container_width=True):
        with st.spinner("AI is preparing your personalised interview questions…"):
            try:
                questions = generate_interview_questions(
                    dict(st.session_state),
                    _ss("target_jd"),
                    _ss("domain") or "Generic Professional",
                    _ss("sub_domain"),
                )
                st.session_state["interview_questions"] = questions
                st.session_state["interview_feedback"] = None
                st.rerun()
            except Exception as exc:
                st.error(f"AI error: {exc}")
                logger.error(traceback.format_exc())

    questions: list[dict] = _ss("interview_questions") or []

    if not questions:
        st.info("💡 Click the button above to generate your personalised interview questions.")
        return

    st.divider()

    # ── Display questions grouped by type ───────────────────────────────────
    tech_qs  = [q for q in questions if q.get("type") == "Technical"]
    hr_qs    = [q for q in questions if q.get("type") == "HR"]
    behav_qs = [q for q in questions if q.get("type") == "Behavioral"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔧 Technical**")
        for i, q in enumerate(tech_qs, 1):
            st.markdown(f"{i}. {q['question']}")
    with col2:
        st.markdown("**🤝 HR**")
        for i, q in enumerate(hr_qs, 1):
            st.markdown(f"{i}. {q['question']}")
    with col3:
        st.markdown("**⭐ Behavioral**")
        for i, q in enumerate(behav_qs, 1):
            st.markdown(f"{i}. {q['question']}")

    st.divider()

    # ── Practice Section ────────────────────────────────────────────────────
    st.markdown('<div class="wizard-section-header">💬 Practice Your Answer</div>', unsafe_allow_html=True)

    q_labels = [f"[{q['type']}]  {q['question']}" for q in questions]
    selected_idx = st.selectbox(
        "Pick a question to practise:",
        options=range(len(q_labels)),
        format_func=lambda i: q_labels[i],
    )

    selected_question = questions[selected_idx]["question"]

    practice_answer = st.text_area(
        "Your Answer",
        height=160,
        placeholder=(
            "Use the STAR method:\n"
            "Situation → Task → Action → Result\n\n"
            "Example start: 'In my previous role at XYZ, I was tasked with…'"
        ),
        key="practice_answer_input",
    )

    if st.button("✨ Evaluate My Answer", type="primary", use_container_width=False):
        if not practice_answer.strip():
            st.warning("Please type your answer first.")
        else:
            with st.spinner("AI coach is reviewing your answer…"):
                try:
                    feedback = evaluate_answer(
                        selected_question,
                        practice_answer,
                        _ss("target_jd"),
                        _ss("domain") or "Generic Professional",
                        _ss("sub_domain"),
                    )
                    st.session_state["interview_feedback"] = feedback
                except Exception as exc:
                    st.error(f"AI error: {exc}")
                    logger.error(traceback.format_exc())

    # ── Feedback display ────────────────────────────────────────────────────
    feedback = _ss("interview_feedback")
    if feedback:
        st.divider()
        score = feedback.get("score", 0)
        colour_class = (
            "ats-score-green"  if score >= 7
            else "ats-score-orange" if score >= 4
            else "ats-score-red"
        )
        st.markdown(
            f'<div class="ats-score-big {colour_class}">{score} / 10</div>',
            unsafe_allow_html=True,
        )
        st.progress(score / 10)

        interpretation = (
            "🟢 Strong answer! You're well-prepared for this question." if score >= 7
            else "🟡 Decent attempt. Apply the suggestions below to sharpen it." if score >= 4
            else "🔴 Needs work. Focus on adding specific examples and measurable results."
        )
        st.info(interpretation)

        st.markdown("**📋 Coach Feedback**")
        st.markdown(feedback.get("feedback", ""))

        hint = feedback.get("ideal_answer_hint", "")
        if hint:
            with st.expander("💡 What a strong answer looks like"):
                st.markdown(hint)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _render_sidebar()

    st.title("📄 AI Resume Builder — India")
    st.markdown(
        "A next-level, AI-powered resume builder tailored for the "
        "**Indian job market** 🇮🇳.  Paste your Job Description, fill in your "
        "details, and let Groq AI craft powerful, ATS-optimised bullet points."
    )
    st.divider()

    # ── Onboarding gate ──────────────────────────────────────────────────────
    _render_onboarding()

    if not st.session_state["onboarding_done"]:
        return  # Don't show tabs until onboarding is complete

    # ── Main wizard tabs ─────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "👤 Personal",
        "🎓 Education",
        "💼 Experience & Projects",
        "🛠 Skills",
        "🤖 ATS Score",
        "📥 Preview & Download",
        "🎯 Interview Prep",
    ])

    with tab1:
        _tab_personal()
    with tab2:
        _tab_education()
    with tab3:
        _tab_experience()
    with tab4:
        _tab_skills()
    with tab5:
        _tab_ats()
    with tab6:
        _tab_preview()
    with tab7:
        _tab_interview_prep()


if __name__ == "__main__":
    main()