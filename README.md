# 🤖 AI Resume Builder

### AI-Powered Resume Builder with ATS Scoring, Resume Parsing, Keyword Optimization & PDF Generation using Groq (Llama 3)

<p align="center">

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-success?style=for-the-badge)](https://ai-resume-builder-4hbg7gvcuqrmddtt79zskj.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama_3-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

</p>

---

> **A production-ready AI Resume Builder that helps users create ATS-friendly resumes, optimize keywords, improve content using AI, and export professional PDF resumes.**

Built using **Python, Streamlit, Groq (Llama 3), Jinja2, WeasyPrint and PDF Processing Libraries.**

---

# 🌐 Live Demo

### 🚀 Try it here

https://ai-resume-builder-4hbg7gvcuqrmddtt79zskj.streamlit.app/

No installation required.

---

# 📸 Application Preview

## Home Page

<img width="1920" height="1020" alt="Screenshot 2026-06-26 144424" src="https://github.com/user-attachments/assets/98188d06-9f8a-45fa-bb67-89a03142bcd6" />



# 📖 Overview

AI Resume Builder is an intelligent web application that leverages **Large Language Models (LLMs)** to generate professional resumes in minutes.

The application helps users:

- Upload an existing resume
- Analyze ATS compatibility
- Improve resume bullet points
- Generate professional summaries
- Optimize keywords
- Compare resume against job descriptions
- Export ATS-friendly PDF resumes

The project is designed primarily for the **Indian job market** while supporting globally accepted resume formats.

---

# ✨ Features

- 🤖 AI Resume Parsing
- 📄 ATS Score Calculation
- 🎯 Keyword Optimization
- 🧠 AI Professional Summary
- 💼 Resume Bullet Enhancement
- 📊 Skill Gap Analysis
- 📑 Multiple Resume Templates
- 📤 Resume Upload
- 📥 PDF Export
- ⚡ Groq Llama 3 Integration
- 🌍 ATS Friendly Resume
- 🇮🇳 Indian Resume Format
- 🎨 Modern Streamlit UI

---

# 🏗 Project Architecture

```text
ai_resume_builder/
│
├── app.py
├── llm_engine.py
├── pdf_generator.py
├── utils.py
│
├── templates/
│   ├── indian_modern.html
│   └── ats_friendly.html
│
├── assets/
│   ├── screenshots/
│   └── demo.gif
│
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

---

# 📦 Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Framework | Streamlit |
| LLM | Groq (Llama 3) |
| Prompt Engineering | Custom Prompt Templates |
| Resume Templates | HTML + Jinja2 |
| PDF Generation | WeasyPrint |
| Resume Parsing | pdfplumber, PyPDF2 |
| Image Processing | Pillow |
| Deployment | Streamlit Cloud |

---

# 🤖 AI Features

| Feature | Description |
|----------|-------------|
| Resume Parsing | Extracts resume into structured JSON |
| ATS Score | Calculates ATS compatibility |
| Keyword Optimization | Adds missing keywords |
| Resume Enhancement | Improves bullet points using STAR method |
| Professional Summary | Generates recruiter-friendly summaries |
| Skill Gap Analysis | Compares resume with job descriptions |

---

# 🎨 Resume Templates

| Template | Best For | ATS Friendly | Photo |
|-----------|----------|-------------|-------|
| Indian Modern | Campus Placements | ✅ | ✅ |
| ATS Friendly | FAANG / Startups | ✅ | ❌ |

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Sudhanshuraj1037/ai-resume-builder.git

cd ai-resume-builder
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create a `.env` file.

Add your Groq API Key.

```text
GROQ_API_KEY=YOUR_API_KEY
```

---

## Run Application

```bash
streamlit run app.py
```

Application will run at

```
http://localhost:8501
```

---

# 📊 Project Highlights

✔ AI-powered Resume Enhancement

✔ ATS Score Analysis

✔ Resume Parsing

✔ Keyword Optimization

✔ Skill Gap Detection

✔ Multiple Resume Templates

✔ PDF Resume Generation

✔ Fast Groq LLM Integration

---

# 📂 Future Improvements

- Cover Letter Generator
- LinkedIn Profile Import
- Resume Version History
- AI Interview Question Generator
- Portfolio Website Generator
- Resume Translation
- Multi-language Support
- One-click Job Description Matching

---

# 📄 License

This project is licensed under the **MIT License**.

Feel free to use it for learning, research and academic purposes.

---

# 👨‍💻 Author

## Sudhanshu Raj

**B.Tech Computer Science & Engineering**

Lovely Professional University

### 🌐 Connect with me

- LinkedIn:
  https://linkedin.com/in/sudhanshu-raj-7b1651321

---

# ⭐ Support

If you found this project useful,

⭐ Star this repository

🍴 Fork it

💡 Share your feedback

Happy Coding! 🚀
