# ================= NLP ATS ANALYZER =================
import re
import nltk
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

nltk.download('punkt')

TECH_KEYWORDS = [
    "python","java","machine learning","data science","docker",
    "aws","nlp","deep learning","sql","mongodb","opencv",
    "tensorflow","pytorch","flask","streamlit","rest api"
]

ACTION_VERBS = [
    "developed","implemented","designed","built","optimized",
    "engineered","created","analyzed","improved","led"
]

# ---------- Text Cleaning ----------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s%+]', '', text)
    return text

# ---------- Skill Extraction ----------
def extract_skills(text):
    found = []
    for skill in TECH_KEYWORDS:
        if skill in text:
            found.append(skill)
    return found

# ---------- Action Verb Detection ----------
def detect_action_verbs(text):
    sentences = nltk.sent_tokenize(text.lower())
    found = []

    for sent in sentences:
        for verb in ACTION_VERBS:
            if sent.strip().startswith(verb):
                found.append(sent)
    return found

# ---------- Quantified Achievement Detection ----------
def detect_quantified(text):
    return re.findall(r"\d+%|\d+\+", text)

# ---------- Readability Score ----------
def readability_score(text):
    try:
        return textstat.flesch_reading_ease(text)
    except:
        return 0

# ---------- TF-IDF Job Similarity ----------
def keyword_similarity(resume_text):

    corpus = [resume_text, " ".join(TECH_KEYWORDS)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return similarity * 100

# ---------- MAIN NLP ATS ----------
def advanced_ats_analysis(summary, skills, projects, internships, achievements):

    full_text = " ".join([summary, skills, projects, internships, achievements])
    clean_text = preprocess(full_text)

    score = 0
    suggestions = []

    # ----- Keyword Similarity -----
    sim_score = keyword_similarity(clean_text)
    score += sim_score * 0.3

    # ----- Skill Extraction -----
    skill_hits = extract_skills(clean_text)
    if len(skill_hits) >= 5:
        score += 20
    else:
        suggestions.append("Add more industry relevant technical skills.")

    # ----- Action Verbs -----
    verbs = detect_action_verbs(clean_text)
    if verbs:
        score += 20
    else:
        suggestions.append("Start bullet points with strong action verbs.")

    # ----- Quantified Achievements -----
    quant = detect_quantified(clean_text)
    if quant:
        score += 15
    else:
        suggestions.append("Add quantified achievements (Ex: Improved accuracy by 20%).")

    # ----- Readability -----
    read_score = readability_score(full_text)
    if read_score > 40:
        score += 15
    else:
        suggestions.append("Improve readability and avoid complex long sentences.")

    return int(min(score,100)), skill_hits, verbs, quant, suggestions
# ================= JOB DESCRIPTION MATCHER =================

def job_description_analysis(jd_text, summary, skills, projects, internships, achievements):

    resume_text = " ".join([summary, skills, projects, internships, achievements])

    jd_clean = preprocess(jd_text)
    resume_clean = preprocess(resume_text)

    # ---------- TF-IDF Similarity ----------
    corpus = [resume_clean, jd_clean]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)

    match_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

    # ---------- Keyword Extraction ----------
    jd_keywords = extract_skills(jd_clean)
    resume_keywords = extract_skills(resume_clean)

    missing_keywords = list(set(jd_keywords) - set(resume_keywords))

    # ---------- Suggestion Generator ----------
    suggestions = []
    if missing_keywords:
        suggestions.append("Consider adding projects or experience related to missing skills.")

    if match_score < 60:
        suggestions.append("Resume is weakly aligned with job description. Add relevant technical content.")

    # ---------- AI Summary Rewriter ----------
    rewritten_summary = summary

    if jd_keywords:
        rewritten_summary = (
            f"{name} is a {degree} student skilled in {', '.join(jd_keywords[:5])}. "
            f"Interested in roles focusing on {', '.join(jd_keywords[:3])}."
        )

    return int(match_score), missing_keywords, suggestions, rewritten_summary

# ================= SMART SKILL RECOMMENDER =================
ROLE_SKILL_MAP = {

    "computer": ["docker", "kubernetes", "fastapi", "system design", "microservices"],

    "ai": ["deep learning", "pytorch", "tensorflow", "mlops", "model deployment"],

    "machine learning": ["mlops", "feature engineering", "model deployment", "pytorch", "scikit-learn"],

    "data": ["sql", "power bi", "tableau", "data engineering", "etl pipelines"],

    "data science": ["statistics", "machine learning", "python", "data visualization", "feature engineering"],

    "software": ["rest api", "microservices", "fastapi", "docker", "system design"],

    "web": ["react", "node.js", "mongodb", "rest api", "graphql"],

    "full stack": ["react", "node.js", "mongodb", "docker", "microservices"],

    "backend": ["fastapi", "spring boot", "microservices", "postgresql", "redis"],

    "frontend": ["react", "typescript", "tailwind css", "next.js", "ui/ux design"],

    "cloud": ["aws", "azure", "gcp", "kubernetes", "terraform"],

    "devops": ["docker", "kubernetes", "terraform", "ci/cd", "jenkins"],

    "cyber": ["network security", "penetration testing", "ethical hacking", "siem tools"],

    "mobile": ["flutter", "react native", "kotlin", "swift", "firebase"],

    "iot": ["embedded systems", "raspberry pi", "arduino", "mqtt", "edge computing"]
}

TRENDING_SKILLS = [

    # Cloud & DevOps
    "docker",
    "kubernetes",
    "terraform",
    "cloud computing",
    "serverless architecture",

    # Backend / API
    "fastapi",
    "microservices",
    "graphql",
    "system design",

    # AI / ML
    "mlops",
    "generative ai",
    "llm integration",
    "rag architecture",

    # Data Engineering
    "data pipelines",
    "apache spark",
    "kafka",
    "airflow",

    # Frontend
    "next.js",
    "typescript",
    "tailwind css",

    # Security
    "zero trust security",
    "penetration testing",

    # Performance & Scaling
    "distributed systems",
    "high availability architecture"
]
CAREER_MAP = {

    # ---------- AI / ML ----------
    "ai": "AI Engineer",
    "machine learning": "Machine Learning Engineer",
    "deep learning": "Deep Learning Engineer",
    "nlp": "NLP Engineer",
    "computer vision": "Computer Vision Engineer",
    "llm": "LLM Engineer",
    "generative ai": "Generative AI Engineer",

    # ---------- Data ----------
    "data": "Data Scientist",
    "analytics": "Data Analyst",
    "data engineering": "Data Engineer",
    "big data": "Big Data Engineer",
    "business intelligence": "BI Developer",

    # ---------- Software ----------
    "software": "Software Engineer",
    "backend": "Backend Developer",
    "frontend": "Frontend Developer",
    "full stack": "Full Stack Developer",
    "system design": "System Architect",

    # ---------- Cloud / DevOps ----------
    "cloud": "Cloud Engineer",
    "devops": "DevOps Engineer",
    "site reliability": "Site Reliability Engineer",
    "kubernetes": "Cloud Native Engineer",

    # ---------- Cybersecurity ----------
    "cyber": "Cyber Security Analyst",
    "ethical hacking": "Ethical Hacker",
    "network security": "Security Engineer",
    "digital forensics": "Digital Forensics Analyst",

    # ---------- Mobile ----------
    "android": "Android Developer",
    "ios": "iOS Developer",
    "flutter": "Mobile App Developer",
    "react native": "Mobile App Developer",

    # ---------- IoT / Embedded ----------
    "iot": "IoT Engineer",
    "embedded": "Embedded Systems Engineer",
    "robotics": "Robotics Engineer",

    # ---------- Blockchain ----------
    "blockchain": "Blockchain Developer",
    "web3": "Web3 Developer",

    # ---------- AR / VR ----------
    "ar": "AR Developer",
    "vr": "VR Developer",
    "metaverse": "Metaverse Developer",

    # ---------- Game ----------
    "game": "Game Developer",
    "unity": "Unity Game Developer",
    "unreal": "Unreal Engine Developer",

    # ---------- UI / UX ----------
    "ui": "UI Designer",
    "ux": "UX Designer",
    "product design": "Product Designer",

    # ---------- Management ----------
    "product": "Product Manager",
    "project": "Project Manager",
    "business": "Business Analyst",

    # ---------- Hardware ----------
    "vlsi": "VLSI Engineer",
    "electronics": "Hardware Engineer"
}
CAREER_ROADMAP = {

    # ---------- AI / ML ----------
    "AI Engineer": [
        "Python Programming",
        "Machine Learning",
        "Deep Learning",
        "PyTorch / TensorFlow",
        "Model Deployment",
        "MLOps"
    ],

    "Machine Learning Engineer": [
        "Statistics & Probability",
        "Scikit-learn",
        "Feature Engineering",
        "Model Optimization",
        "ML Deployment"
    ],

    "Deep Learning Engineer": [
        "Neural Networks",
        "CNN / RNN",
        "PyTorch / TensorFlow",
        "GPU Computing",
        "Advanced AI Architectures"
    ],

    "NLP Engineer": [
        "Text Processing",
        "Transformers",
        "HuggingFace",
        "LLM Fine-tuning",
        "RAG Systems"
    ],

    "Computer Vision Engineer": [
        "OpenCV",
        "Image Processing",
        "Object Detection",
        "YOLO / Vision Transformers",
        "Edge AI Deployment"
    ],

    "Generative AI Engineer": [
        "LLM Integration",
        "Prompt Engineering",
        "RAG Architecture",
        "LangChain",
        "AI Agents"
    ],

    # ---------- Data ----------
    "Data Scientist": [
        "Python",
        "Statistics",
        "SQL",
        "Machine Learning",
        "Data Visualization"
    ],

    "Data Analyst": [
        "Excel",
        "SQL",
        "Power BI / Tableau",
        "Data Cleaning",
        "Business Insights"
    ],

    "Data Engineer": [
        "SQL",
        "ETL Pipelines",
        "Apache Spark",
        "Kafka",
        "Data Warehousing"
    ],

    "Big Data Engineer": [
        "Hadoop",
        "Spark",
        "Kafka",
        "Distributed Systems",
        "Data Lakes"
    ],

    # ---------- Software ----------
    "Software Engineer": [
        "Data Structures",
        "Algorithms",
        "System Design",
        "OOP",
        "Software Architecture"
    ],

    "Backend Developer": [
        "FastAPI / Spring Boot",
        "Database Design",
        "Microservices",
        "System Design",
        "Caching"
    ],

    "Frontend Developer": [
        "HTML / CSS",
        "JavaScript",
        "React / Angular",
        "TypeScript",
        "UI Performance Optimization"
    ],

    "Full Stack Developer": [
        "Frontend Frameworks",
        "Backend APIs",
        "Database Management",
        "Authentication",
        "Deployment"
    ],

    # ---------- Cloud / DevOps ----------
    "Cloud Engineer": [
        "AWS / Azure / GCP",
        "Docker",
        "Kubernetes",
        "Terraform",
        "Cloud Security"
    ],

    "DevOps Engineer": [
        "CI/CD Pipelines",
        "Docker",
        "Kubernetes",
        "Monitoring Tools",
        "Infrastructure Automation"
    ],

    "Site Reliability Engineer": [
        "Monitoring Systems",
        "Automation",
        "Incident Response",
        "Cloud Infrastructure",
        "Performance Tuning"
    ],

    # ---------- Cybersecurity ----------
    "Cyber Security Analyst": [
        "Network Security",
        "Penetration Testing",
        "SIEM Tools",
        "Threat Analysis",
        "Security Auditing"
    ],

    "Ethical Hacker": [
        "Penetration Testing",
        "Exploit Development",
        "Bug Bounty Hunting",
        "Network Security",
        "Reverse Engineering"
    ],

    # ---------- Mobile ----------
    "Mobile App Developer": [
        "Flutter / React Native",
        "UI Design",
        "API Integration",
        "App Optimization",
        "App Store Deployment"
    ],

    "Android Developer": [
        "Kotlin / Java",
        "Android SDK",
        "Jetpack Compose",
        "Firebase Integration",
        "App Publishing"
    ],

    "iOS Developer": [
        "Swift",
        "iOS SDK",
        "UI Kit",
        "Core Data",
        "App Store Deployment"
    ],

    # ---------- IoT ----------
    "IoT Engineer": [
        "Embedded Systems",
        "Arduino / Raspberry Pi",
        "MQTT Protocol",
        "Edge Computing",
        "IoT Cloud Integration"
    ],

    "Robotics Engineer": [
        "ROS",
        "Embedded Programming",
        "Control Systems",
        "Sensor Fusion",
        "AI Robotics"
    ],

    # ---------- Blockchain ----------
    "Blockchain Developer": [
        "Solidity",
        "Smart Contracts",
        "Ethereum",
        "Web3.js",
        "Blockchain Security"
    ],

    # ---------- AR/VR ----------
    "AR Developer": [
        "Unity",
        "ARKit / ARCore",
        "3D Modeling",
        "Spatial Computing"
    ],

    "VR Developer": [
        "Unity / Unreal",
        "3D Rendering",
        "VR Interaction Design",
        "Game Physics"
    ],

    # ---------- Game ----------
    "Game Developer": [
        "Unity / Unreal Engine",
        "C# / C++",
        "Game Physics",
        "Multiplayer Networking"
    ],

    # ---------- UI/UX ----------
    "UI Designer": [
        "Figma",
        "Design Principles",
        "Typography",
        "Color Theory",
        "Responsive Design"
    ],

    "UX Designer": [
        "User Research",
        "Wireframing",
        "Usability Testing",
        "Interaction Design"
    ],

    # ---------- Management ----------
    "Product Manager": [
        "Product Strategy",
        "Market Research",
        "Agile Methodology",
        "User Analytics"
    ],

    "Business Analyst": [
        "Data Analysis",
        "Requirement Gathering",
        "Process Modeling",
        "Stakeholder Communication"
    ],

    # ---------- Hardware ----------
    "VLSI Engineer": [
        "Digital Electronics",
        "Verilog / VHDL",
        "Chip Design",
        "FPGA Development"
    ],

    "Hardware Engineer": [
        "Circuit Design",
        "PCB Design",
        "Embedded Programming",
        "Signal Processing"
    ]
}
def predict_career(degree, skills):

    text = preprocess(degree + " " + skills)

    for key in CAREER_MAP:
        if key in text:
            return CAREER_MAP[key]

    return "Software Engineer"
def career_path_from_role(role):

    skills = CAREER_ROADMAP.get(role, [])

    return skills

import matplotlib.pyplot as plt
import random

def generate_demand_graph(role):

    years = ["2024", "2025", "2026", "2027", "2028"]

    demand = [random.randint(60, 90) for _ in years]

    fig, ax = plt.subplots()

    ax.plot(years, demand)
    ax.set_title(f"Future Demand Trend for {role}")
    ax.set_ylabel("Demand Index")

    return fig



def recommend_skills(degree, projects, current_skills, github):

    recommendations = set()

    degree_text = preprocess(degree)
    project_text = preprocess(projects)
    skill_text = preprocess(current_skills)

    # ---------- Degree Based ----------
    for key in ROLE_SKILL_MAP:
        if key in degree_text:
            recommendations.update(ROLE_SKILL_MAP[key])

    # ---------- Project Based ----------
    if "machine learning" in project_text:
        recommendations.update(["mlops", "pytorch", "model deployment"])

    if "web" in project_text or "app" in project_text:
        recommendations.update(["fastapi", "microservices"])

    if "cloud" in project_text:
        recommendations.update(["kubernetes", "docker"])

    # ---------- GitHub Based ----------
    if github:
        recommendations.update(["ci/cd", "open source contribution"])

    # ---------- Trending Skills ----------
    recommendations.update(TRENDING_SKILLS[:2])

    # ---------- Remove Existing Skills ----------
    existing = set(skill_text.split())
    final_recommendations = [s for s in recommendations if s not in existing]

    return final_recommendations[:6]


# ================= AI MOCK INTERVIEW =================
import requests
import speech_recognition as sr

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"


def ollama_generate(prompt):

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]

    except:
        return "AI service unavailable"


# ---------- Generate Interview Questions ----------
def generate_interview_questions(role):

    prompt = f"""
    Generate mock interview questions for {role}.
    Provide:
    1. 3 Technical Questions
    2. 2 HR Questions
    3. 1 Coding Question
    """

    return ollama_generate(prompt)


# ---------- Answer Feedback ----------
def evaluate_answer(question, answer):

    prompt = f"""
    Interview Question: {question}
    Candidate Answer: {answer}

    Give:
    - Strengths
    - Weaknesses
    - Improvement Suggestions
    """

    return ollama_generate(prompt)


# ---------- Voice Input ----------
def speech_to_text():

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:

            st.info("🎤 Listening... Speak now")

            recognizer.adjust_for_ambient_noise(source, duration=1)

            audio = recognizer.listen(
                source,
                timeout=5,            # wait max 5 sec for speech
                phrase_time_limit=10  # record max 10 sec
            )

        st.success("Processing speech...")

        text = recognizer.recognize_google(audio)
        return text

    except sr.WaitTimeoutError:
        return "⚠️ No speech detected. Try again."

    except sr.UnknownValueError:
        return "⚠️ Could not understand audio."

    except sr.RequestError:
        return "⚠️ Speech service unavailable."

    except Exception as e:
        return f"Error: {str(e)}"

# ---------- Generate AI Solution ----------
def generate_solution(question):

    prompt = f"""
    Provide a clear, professional model answer for this interview question:

    Question: {question}

    Keep it structured and concise.
    """

    return ollama_generate(prompt)


import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
import os
import tempfile
import html
from datetime import datetime

st.set_page_config(page_title="Resume Starter", layout="centered")

# ----------------- Utility functions -----------------
def hex_to_rgb(hex_color):
    """Convert hex color (#rrggbb) to an (r,g,b) tuple of ints 0-255."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def safe_text(s):
    return "" if s is None else str(s)


def bullet_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]

#
# ----------------- Sidebar: theme & options -----------------
st.sidebar.title("Template Settings")
accent_color = st.sidebar.color_picker("Accent color", "#1f77b4")
template = st.sidebar.selectbox("Template", ["Professional (Classic)", "Modern (Card)", "Minimal"])

st.sidebar.markdown(
    """
Small tips:
- Use commas to separate multiple items (projects, internships, skills).
- Upload a square-ish profile photo for best results.
"""
)

# ----------------- Inject base CSS for the Streamlit app -----------------
base_css = f"""
<style>
/* Page background and card */
* {{ box-sizing: border-box; }}
section.main .block-container {{
  max-width: 900px;
  padding-top: 1rem;
  padding-bottom: 3rem;
}}
.stButton>button {{
  background: linear-gradient(90deg, {accent_color}, #0b6cb1);
  color: white;
  border: none;
}}
.resume-meta {{
  display:flex;
  gap:10px;
  align-items:center;
  color:#333;
}}
.small-muted {{
  color: #666;
  font-size: 0.9rem;
}}
.resume-preview {{
  border:1px solid rgba(0,0,0,0.08);
  border-radius:8px;
  padding:16px;
  background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(250,250,250,0.98));
}}
.preview-header {{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:10px;
}}
.preview-name {{
  font-size:1.6rem;
  font-weight:700;
  color:{accent_color};
}}
.preview-section-title {{
  color:{accent_color};
  margin-top:8px;
  margin-bottom:4px;
  font-weight:600;
}}
@media (max-width: 600px) {{
  .preview-name {{ font-size:1.2rem; }}
}}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

# ----------------- Inputs -----------------
st.title("📄 Resume Starter for Freshers")

# Profile photo
photo = st.file_uploader("Upload Profile Photo (Optional) — jpg, png, jpeg", type=["jpg", "png", "jpeg"])

# Basic input
c1, c2 = st.columns([2, 1])
with c1:
    name = st.text_input("Full Name", "")
    email = st.text_input("Email", "")
    phone = st.text_input("Phone Number", "")
with c2:
    linkedin = st.text_input("LinkedIn ID", "")
    github = st.text_input("Github ID", "")

# Summary with char count
summary = st.text_area("Professional Summary (short, 2-4 lines)", height=120)
summary_chars = len(summary)
st.markdown(f"<div class='small-muted'>Summary length: {summary_chars} characters</div>", unsafe_allow_html=True)

# Education
st.subheader("Education")
college = st.text_input("College / University", "")
degree = st.text_input("Degree / Department", "")
cgpa = st.text_input("CGPA", "")
batch = st.text_input("Batch", "")
hsc = st.text_input("HSC %", "")
sslc = st.text_input("SSLC %", "")

# Other details
st.subheader("Other Details")
internships = st.text_area("Internships (comma separated)", "")
projects = st.text_area("Projects (comma separated)", "")
certifications = st.text_area("Certifications (comma separated)", "")
achievements = st.text_area("Achievements (comma separated)", "")
skills = st.text_area("Skills (comma separated)", "")
clubs = st.text_input("Clubs / Activities", "")
leetcode = st.text_input("Leetcode / Coding Achievements", "")

# ----------------- JOB DESCRIPTION INPUT -----------------
st.subheader("🎯 Job Description Matcher")

job_description = st.text_area(
    "Paste Job Description Here",
    height=200,
    placeholder="Paste company job role description here..."
)

# LinkedIn about generator
def generate_linkedin_about():
    # Escape the inputs to avoid injecting HTML into the preview
    esc_name = html.escape(name)
    esc_degree = html.escape(degree)
    esc_college = html.escape(college)
    esc_skills = html.escape(skills)
    esc_projects = html.escape(projects)
    about = f"{esc_name} is a {esc_degree} student at {esc_college}. Strong interest in {esc_skills}. Worked on projects like {esc_projects}."
    return about

# ----------------- HTML Resume Preview (client-side) -----------------
def render_preview():

    preview_html = f"""
<style>
.resume-modern {{
    font-family: 'Segoe UI', sans-serif;
    background:white;
    padding:24px;
    border-radius:10px;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}}

.resume-header {{
    display:grid;
    grid-template-columns: 1fr 110px;
    gap:20px;
    align-items:center;
    border-bottom:2px solid {accent_color};
    padding-bottom:12px;
}}

.resume-name {{
    font-size:26px;
    font-weight:700;
    color:{accent_color};
}}

.resume-meta {{
    font-size:14px;
    color:#555;
    margin-top:4px;
}}

.resume-grid {{
    display:grid;
    grid-template-columns: 35% 65%;
    gap:20px;
    margin-top:20px;
}}

.left-col {{
    background:#f7f9fc;
    padding:14px;
    border-radius:8px;
}}

.section-title {{
    font-weight:700;
    color:{accent_color};
    margin-top:12px;
    margin-bottom:6px;
    font-size:15px;
}}

.resume-photo {{
    width:100px;
    height:100px;
    object-fit:cover;
    border-radius:8px;
    border:2px solid {accent_color};
}}

ul {{
    padding-left:18px;
    margin:4px 0;
}}

li {{
    margin-bottom:4px;
}}

</style>

<div class="resume-modern" id="resume-preview">

<div class="resume-header">

<div>
<div class="resume-name">{html.escape(name or "Your Name")}</div>
<div class="resume-meta">{html.escape(degree)} • {html.escape(college)}</div>
<div class="resume-meta">{html.escape(phone)} | {html.escape(email)}</div>
</div>
"""

    # -------- Photo ----------
    if photo:
        import base64
        raw = photo.read()
        b64 = base64.b64encode(raw).decode()
        preview_html += f'<img src="data:image/png;base64,{b64}" class="resume-photo"/>'
        try:
            photo.seek(0)
        except:
            pass

    preview_html += "</div>"  # header end

    # -------- Body Layout ----------
    preview_html += '<div class="resume-grid">'

    # ===== LEFT COLUMN =====
    preview_html += '<div class="left-col">'

    if skills:
        preview_html += "<div class='section-title'>Skills</div><ul>"
        for s in bullet_list(skills):
            preview_html += f"<li>{html.escape(s)}</li>"
        preview_html += "</ul>"

    if certifications:
        preview_html += "<div class='section-title'>Certifications</div><ul>"
        for c in bullet_list(certifications):
            preview_html += f"<li>{html.escape(c)}</li>"
        preview_html += "</ul>"

    if achievements:
        preview_html += "<div class='section-title'>Achievements</div><ul>"
        for a in bullet_list(achievements):
            preview_html += f"<li>{html.escape(a)}</li>"
        preview_html += "</ul>"

    preview_html += f"""
    <div class='section-title'>Links</div>
    <div>LinkedIn: {html.escape(linkedin or "-")}</div>
    <div>GitHub: {html.escape(github or "-")}</div>
    """

    preview_html += "</div>"  # left end

    # ===== RIGHT COLUMN =====
    preview_html += "<div>"

    if summary:
        preview_html += f"<div class='section-title'>Professional Summary</div><p>{html.escape(summary)}</p>"

    preview_html += f"""
    <div class='section-title'>Education</div>
    <p>{html.escape(degree)}<br>
    {html.escape(college)}<br>
    CGPA: {html.escape(cgpa)} | Batch: {html.escape(batch)}</p>
    """

    if projects:
        preview_html += "<div class='section-title'>Projects</div><ul>"
        for p in bullet_list(projects):
            preview_html += f"<li>{html.escape(p)}</li>"
        preview_html += "</ul>"

    if internships:
        preview_html += "<div class='section-title'>Internships</div><ul>"
        for i in bullet_list(internships):
            preview_html += f"<li>{html.escape(i)}</li>"
        preview_html += "</ul>"

    preview_html += "</div>"  # right end
    preview_html += "</div>"  # grid end

    # -------- Buttons ----------
    preview_html += f"""
<div style="margin-top:20px;">
<button onclick="window.print()" style="background:{accent_color};color:white;padding:8px 14px;border:none;border-radius:6px;">
Print / Save PDF
</button>
</div>
"""

    preview_html += "</div>"
    return preview_html



# Show LinkedIn about and preview area
st.subheader("Preview")
preview_html = render_preview()
components.html(preview_html, height=520, scrolling=True)

if st.button("🧠 Run Advanced NLP ATS Analysis"):

    ats_score, skill_hits, verbs, quant, suggestions = advanced_ats_analysis(
        summary, skills, projects, internships, achievements
    )

    st.session_state.ats_score = ats_score
    st.session_state.skill_hits = skill_hits
    st.session_state.verbs = verbs
    st.session_state.quant = quant
    st.session_state.suggestions = suggestions
    
if "ats_score" in st.session_state:

    st.subheader("📊 NLP ATS Score")
    st.progress(st.session_state.ats_score/100)
    st.write(f"### Score: {st.session_state.ats_score}%")

    st.subheader("🛠 Skills Detected")
    st.write(", ".join(st.session_state.skill_hits) if st.session_state.skill_hits else "No skills detected")

    st.subheader("🚀 Action Verb Sentences")
    for v in st.session_state.verbs[:5]:
        st.write("✔", v)

    st.subheader("📈 Quantified Achievements Found")
    st.write(", ".join(st.session_state.quant) if st.session_state.quant else "None detected")

    st.subheader("⚠ Improvement Suggestions")
    for s in st.session_state.suggestions:
        st.write("•", s)


# ================= JOB DESCRIPTION ANALYZER =================
if st.button("🎯 Match Resume with Job Description"):

    if not job_description.strip():
        st.warning("Please paste Job Description first")
    else:

        match_score, missing_keywords, jd_suggestions, new_summary = job_description_analysis(
            job_description, summary, skills, projects, internships, achievements
        )

        st.subheader("📊 Resume vs Job Match Score")
        st.progress(match_score/100)
        st.write(f"### Match Score: {match_score}%")

        st.subheader("❌ Missing Keywords")
        if missing_keywords:
            for k in missing_keywords:
                st.write("•", k)
        else:
            st.success("Your resume covers most JD keywords!")

        st.subheader("💡 Suggestions")
        for s in jd_suggestions:
            st.write("•", s)

        st.subheader("✨ AI Rewritten Summary Suggestion")
        st.info(new_summary)


# ================= SMART SKILL RECOMMENDATION =================
if st.button("🧠 Recommend Skills To Learn"):

    recommended = recommend_skills(degree, projects, skills, github)

    st.subheader("🚀 Recommended Skills")

    if recommended:
        for skill in recommended:
            st.write("✔ You should learn:", skill)
    else:
        st.success("Your skill set already looks strong!")
        
# ================= ROLE BASED CAREER PATH =================
st.header("🚀 AI Career Path Predictor")

selected_role = st.selectbox(
    "Select Career Role",
    list(CAREER_ROADMAP.keys())
)
if st.button("📊 Generate Career Path"):

    st.subheader(f"🎯 Career Role: {selected_role}")

    role_skills = career_path_from_role(selected_role)

    # ----- Required Skills -----
    st.subheader("🧩 Required Skills & Learning Roadmap")

    for skill in role_skills:
        st.write("✔", skill)

    # ----- Demand Graph -----
    st.subheader("📈 Future Demand Trend")

    fig = generate_demand_graph(selected_role)
    st.pyplot(fig)

# ================= CAREER PATH PREDICTOR UI =================

st.header("🚀  Career Path Predictor [by Degree & Skills]")

if st.button("🔮 Predict My Career Path"):

    predicted_role = predict_career(degree, skills)

    st.subheader("🎯 Best Career Role")
    st.success(predicted_role)

    # -------- Learning Path --------
    roadmap = CAREER_ROADMAP.get(predicted_role, [])

    st.subheader("📚 Learning Roadmap")

    for step in roadmap:
        st.write("✔", step)

    # -------- Demand Graph --------
    st.subheader("📈 Future Demand Trend")

    fig = generate_demand_graph(predicted_role)
    st.pyplot(fig)

# ================= MOCK INTERVIEW UI =================
st.header("🎤 AI Mock Interview Simulator")

roles = st.multiselect(
    "Select Job Roles",
    [
        "Data Scientist",
        "Machine Learning Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer",
        "AI Engineer"
    ]
)

if st.button("Generate Interview Questions"):

    if roles:

        all_questions = ""

        for role in roles:
            questions = generate_interview_questions(role)
            all_questions += f"\n\n### {role} Interview Questions\n{questions}\n"

        st.session_state.questions = all_questions
        st.markdown(all_questions)

    else:
        st.warning("Please select at least one role")

if "questions" in st.session_state:

    st.subheader("📝 Answer Questions")

    user_question = st.text_area("Paste question here")

    user_answer = st.text_area("Your Answer")

    # -------- Toggle Solution Button --------
    if "show_solution" not in st.session_state:
        st.session_state.show_solution = False

    label = "💡 Show Solution" if not st.session_state.show_solution else "❌ Hide Solution"
    if st.button(label):


        st.session_state.show_solution = not st.session_state.show_solution

        if st.session_state.show_solution and user_question:
            st.session_state.solution = generate_solution(user_question)

    # -------- Display Solution --------
    if st.session_state.get("show_solution", False):
        st.subheader("📘  Suggested Solution")
        st.info(st.session_state.get("solution", "No solution generated"))

    # -------- Evaluate Answer --------
    if st.button("🤖 Evaluate Answer"):

        with st.spinner("🤖 AI Evaluating your answer..."):
            feedback = evaluate_answer(user_question, user_answer)

        st.subheader("📊 Feedback")
        st.write(feedback)
       


# ================= PORTFOLIO WEBSITE GENERATOR =================

def generate_portfolio_html():

    project_list = "".join(
        [f"<li>{html.escape(p)}</li>" for p in bullet_list(projects)]
    )

    skill_list = "".join(
        [f"<span class='skill'>{html.escape(s)}</span>" for s in bullet_list(skills)]
    )

    portfolio_html = f"""
<!DOCTYPE html>
<html>
<head>
<title>{html.escape(name)} Portfolio</title>

<style>
body {{
    font-family: Arial;
    margin:0;
    background:#f4f6f9;
}}

header {{
    background:{accent_color};
    color:white;
    padding:40px;
    text-align:center;
}}

.section {{
    padding:30px;
    margin:20px;
    background:white;
    border-radius:8px;
}}

.skill {{
    display:inline-block;
    background:{accent_color};
    color:white;
    padding:6px 10px;
    margin:4px;
    border-radius:6px;
}}

button {{
    background:{accent_color};
    color:white;
    padding:10px 18px;
    border:none;
    border-radius:6px;
    cursor:pointer;
}}
</style>

</head>

<body>

<header>
<h1>{html.escape(name)}</h1>
<p>{html.escape(degree)} | {html.escape(college)}</p>
<p>{html.escape(email)} | {html.escape(phone)}</p>
</header>

<div class="section">
<h2>About Me</h2>
<p>{html.escape(summary)}</p>
</div>

<div class="section">
<h2>Projects</h2>
<ul>
{project_list}
</ul>
</div>

<div class="section">
<h2>Skills</h2>
{skill_list}
</div>

<div class="section">
<h2>Links</h2>
<p>GitHub: {html.escape(github)}</p>
<p>LinkedIn: {html.escape(linkedin)}</p>
</div>

<div class="section">
<h2>Resume</h2>
<button onclick="alert('Upload your resume PDF here after deployment')">
Download Resume
</button>
</div>

</body>
</html>
"""

    return portfolio_html

# ================= PORTFOLIO WEBSITE BUTTON =================
if st.button("🌐 Generate Portfolio Website"):

    portfolio_html = generate_portfolio_html()

    st.subheader("💻 Portfolio Preview")

    components.html(portfolio_html, height=600, scrolling=True)

    st.download_button(
        "📥 Download Portfolio HTML",
        portfolio_html,
        file_name="portfolio.html",
        mime="text/html"
    )

# ================= AI INTERVIEW QUESTIONS USING OLLAMA =================



       

# ----------------- PDF GENERATOR -----------------
def create_resume_pdf(accent_hex):
    pdf = FPDF()
    pdf.add_page()
    # Fonts: try to add DejaVu if available, otherwise use core fonts
    font_path = os.path.join(os.getcwd(), "DejaVuSans.ttf")
    try:
        if os.path.exists(font_path):
            pdf.add_font("DejaVu", "", font_path, uni=True)
            base_font = "DejaVu"
        else:
            base_font = "Arial"
    except Exception:
        base_font = "Arial"

    # Add photo if provided (save temporary)
    photo_path = None
    if photo is not None:
        try:
            # Save a temporary file
            suffix = ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(photo.read())
            tmp.flush()
            tmp.close()
            photo_path = tmp.name
            # reset file pointer in case Streamlit needs it later
            try:
                photo.seek(0)
            except Exception:
                pass
        except Exception:
            photo_path = None

    # Header
    pdf.set_font(base_font, size=18)
    r, g, b = hex_to_rgb(accent_hex)
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 10, safe_text(name or "Your Name"), ln=True)
    # Contact info
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(base_font, size=11)
    contact = f"{phone} | {email} | LinkedIn: {linkedin or '-'} | GitHub: {github or '-'}"
    pdf.multi_cell(0, 6, contact)
    pdf.ln(2)
    pdf.set_draw_color(r, g, b)
    pdf.set_line_width(0.6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # Photo on top right if exists
    if photo_path:
        try:
            pdf.image(photo_path, x=150, y=15, w=35)
        except Exception:
            pass

    # Summary
    if summary:
        pdf.set_text_color(r, g, b)
        pdf.set_font(base_font, size=13)
        pdf.cell(0, 7, "PROFESSIONAL SUMMARY", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(base_font, size=11)
        pdf.multi_cell(0, 6, summary)
        pdf.ln(2)

    # Education
    pdf.set_text_color(r, g, b)
    pdf.set_font(base_font, size=13)
    pdf.cell(0, 7, "EDUCATION", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(base_font, size=11)
    pdf.multi_cell(0, 6, f"{degree} | CGPA: {cgpa}")
    pdf.multi_cell(0, 6, f"{college} | Batch: {batch}")
    pdf.multi_cell(0, 6, f"HSC: {hsc} | SSLC: {sslc}")
    pdf.ln(2)

    # Sections helper
    def add_section(title, items):
        if not items:
            return
        pdf.set_text_color(r, g, b)
        pdf.set_font(base_font, size=13)
        pdf.cell(0, 7, title, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(base_font, size=11)
        for i, it in enumerate(items, 1):
            pdf.multi_cell(0, 6, f"• {it}")
        pdf.ln(1)

    add_section("INTERNSHIPS", bullet_list(internships))
    add_section("PROJECTS", bullet_list(projects))
    add_section("SKILLS", bullet_list(skills))
    add_section("CERTIFICATIONS", bullet_list(certifications))
    add_section("ACHIEVEMENTS", bullet_list(achievements))

    # Coding achievements
    if leetcode:
        pdf.set_text_color(r, g, b)
        pdf.set_font(base_font, size=13)
        pdf.cell(0, 7, "CODING ACHIEVEMENTS", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(base_font, size=11)
        pdf.multi_cell(0, 6, leetcode)

    # Footer small print
    pdf.set_y(-25)
    pdf.set_font(base_font, size=9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Generated with Resume Starter • {datetime.utcnow().date()}", ln=True, align="C")

    # Output to temporary file
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(out_tmp.name)
    out_tmp.close()

    # cleanup photo temp
    if photo_path:
        try:
            os.remove(photo_path)
        except Exception:
            pass

    return out_tmp.name


# ----------------- Generate Button -----------------
if st.button("Generate Resume + LinkedIn About"):
    if not name:
        st.warning("Enter name first")
    else:
        st.subheader("🌐 LinkedIn About")
        about_text = generate_linkedin_about()
        st.write(about_text)

        st.info("Creating PDF — uses chosen accent color and simple professional layout.")
        pdf_file_path = create_resume_pdf(accent_color)

        with open(pdf_file_path, "rb") as f:
            st.download_button("📥 Download Resume (PDF)", f, file_name="Resume.pdf", mime="application/pdf")

        # Clean up temporary PDF after offering download
        try:
            os.remove(pdf_file_path)
        except Exception:
            pass

