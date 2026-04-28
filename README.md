# ai-resume-builder
AI-powered resume builder with ATS scoring, keyword optimization, and PDF generation using LLM (Groq Llama 3)


# 📄 AI Resume Builder — India
> A next-level, production-ready AI-powered resume builder tailored for the Indian job market.
> Built as a university-level AI project using Streamlit + Groq (Llama-3 70B).

---

## 🏗 Project Architecture

```
ai_resume_builder/
├── app.py                  # Main Streamlit UI — wizard tabs, session state
├── llm_engine.py           # All Groq API calls & prompt engineering
├── pdf_generator.py        # Jinja2 rendering + WeasyPrint PDF conversion
├── utils.py                # PDF parsing, image→Base64, JSON helpers, schema
├── templates/
│   ├── indian_modern.html  # Two-column template with photo (Indian standard)
│   └── ats_friendly.html   # Single-column plain template (ATS optimised)
├── requirements.txt
├── .env.example            # Copy to .env and add your GROQ_API_KEY
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / download the project
```bash
git clone https://github.com/your-username/ai-resume-builder-india.git
cd ai-resume-builder-india
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **WeasyPrint** requires system libraries.  
> Ubuntu/Debian: `sudo apt-get install libpango-1.0-0 libpangoft2-1.0-0`  
> macOS: `brew install pango`  
> Windows: Use pdfkit instead (install `wkhtmltopdf` binary).

### 4. Configure your API key
```bash
cp .env.example .env
# Edit .env and add your Groq API key from https://console.groq.com
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🤖 AI Features

| Feature | Model Temp | Prompt Strategy |
|---|---|---|
| Parse uploaded PDF → JSON | 0.2 | Strict schema extraction, zero hallucination |
| Enhance bullets (STAR method) | 0.7 | Few-shot + domain vocabulary injection |
| Skill gap analysis | 0.3 | JD vs skills set comparison |
| ATS Match Score (0–100) | 0.0 | Chain-of-thought section scoring |
| Professional summary | 0.6 | Role + achievements synthesis |

---

## 🎨 Resume Templates

| Template | Best For | Photo | Columns |
|---|---|---|---|
| Indian Modern | Campus placement, MNCs | ✅ Yes | 2-column |
| ATS Friendly | Job portals, FAANG, startups | ❌ No | Single-column |

---

## 📐 Indian Resume Schema (Key Fields)

- **Personal**: DOB, Gender, Languages Known, Full Address, Photo
- **Education**: 10th / 12th / Diploma / UG / PG with CGPA or Percentage
- **Sections**: Summary, Experience, Projects, Skills, Certifications, Co-Curricular

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key (free at console.groq.com) |

---

## 📦 Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama-3 70B / Mixtral 8×7B fallback)
- **PDF**: WeasyPrint (primary) + pdfkit (fallback)
- **Templates**: Jinja2 HTML
- **Parsing**: pdfplumber + PyPDF2
- **Image**: Pillow → Base64

---

## 🧑‍💻 Contributing

Pull requests welcome!  Please open an issue first to discuss major changes.

---

## 📄 License

MIT License — free to use for academic and personal projects.
