import streamlit as st
import joblib
import pdfplumber
import pandas as pd
import re


# =========================
# Load ML Model
# =========================

model = joblib.load("model/resume_classifier.pkl")
tfidf = joblib.load("model/tfidf.pkl")


# =========================
# Text Cleaning
# =========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# PDF Text Extraction
# =========================

def extract_text_from_pdf(pdf_file):
    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# =========================
# Skill Extraction
# =========================

def extract_skills(text):

    try:
        skills = pd.read_csv(
            "skills.csv",
            header=None
        )[0].tolist()

    except:
        return []

    text = text.lower()

    found_skills = []

    for skill in skills:
        if skill.lower() in text:
            found_skills.append(skill)

    return list(set(found_skills))


# =========================
# ATS Score
# =========================

def calculate_ats_score(
    resume_skills,
    jd_skills
):

    if len(jd_skills) == 0:
        return 0

    matched = len(
        set(resume_skills)
        &
        set(jd_skills)
    )

    score = (
        matched /
        len(jd_skills)
    ) * 100
    
    if score >= 80:
       st.success("🟢 Excellent Match")
    elif score >= 60:
       st.warning("🟡 Moderate Match")
    else:
       st.error("🔴 Low Match")
    return round(score, 2)


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄"
)

st.title("📄 AI Resume Analyzer")

st.write(
    "Upload a resume and compare it with a job description."
)

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description Here",
    height=200
)

# =========================
# Main Logic
# =========================

if uploaded_file:

    try:

        # Extract Resume Text
        resume_text = extract_text_from_pdf(
            uploaded_file
        )

        if len(resume_text.strip()) == 0:
            st.error(
                "Could not extract text from PDF."
            )

        else:

            # Predict Category
            cleaned_text = clean_text(
                resume_text
            )

            vector = tfidf.transform(
                [cleaned_text]
            )

            prediction = model.predict(
                vector
            )[0]

            st.success(
                f"Predicted Resume Category: {prediction}"
            )

            # Extract Resume Skills
            resume_skills = extract_skills(
                resume_text
            )

            st.subheader("Skills Found")

            if resume_skills:
                st.write(resume_skills)
            else:
                st.write(
                    "No skills detected from skills.csv"
                )

            # ATS Score Section
            if job_description:

                jd_skills = extract_skills(
                    job_description
                )

                score = calculate_ats_score(
                    resume_skills,
                    jd_skills
                )

                missing_skills = list(
                    set(jd_skills)
                    -
                    set(resume_skills)
                )

                st.subheader(
                    "ATS Analysis"
                )

                st.metric(
                    "ATS Match Score",
                    f"{score}%"
                )

                st.write(
                    "Job Description Skills:",
                    jd_skills
                )

                st.write(
                    "Missing Skills:",
                    missing_skills
                )

            st.subheader(
                "Extracted Resume Text"
            )

            st.text_area(
                "",
                resume_text[:3000],
                height=300
            )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )