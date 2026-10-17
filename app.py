
import re
from pathlib import Path

import joblib
import networkx as nx
import pandas as pd
import pdfplumber
import plotly.graph_objects as go
import streamlit as st

from career_navigator import recommend_careers, get_career_graph


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer | Career Navigator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

CLASSIFIER_PATH = MODEL_DIR / "resume_classifier.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf.pkl"
SKILLS_PATH = BASE_DIR / "skills.csv"


# ============================================================
# THEME-AWARE PLOTLY COLORS
# No global CSS overrides: Streamlit controls page text colors.
# ============================================================

def get_chart_theme():
    base = st.get_option("theme.base")
    is_dark = base == "dark"

    if is_dark:
        return {
            "template": "plotly_dark",
            "paper": "#111827",
            "plot": "#111827",
            "text": "#F8FAFC",
            "muted": "#CBD5E1",
            "grid": "#374151",
            "matched": "#34D399",
            "missing": "#F87171",
            "primary": "#A78BFA",
            "secondary": "#60A5FA",
        }

    return {
        "template": "plotly_white",
        "paper": "#FFFFFF",
        "plot": "#FFFFFF",
        "text": "#111827",
        "muted": "#475569",
        "grid": "#E2E8F0",
        "matched": "#059669",
        "missing": "#DC2626",
        "primary": "#7C3AED",
        "secondary": "#2563EB",
    }


COLORS = get_chart_theme()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found: {CLASSIFIER_PATH}"
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {VECTORIZER_PATH}"
        )

    classifier = joblib.load(CLASSIFIER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return classifier, vectorizer


# ============================================================
# TEXT AND PDF UTILITIES
# ============================================================

def clean_text(text):
    """Normalize text for classification and comparison."""
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    return text.strip()


def extract_text_from_pdf(uploaded_file):
    """Extract readable text from an uploaded PDF."""
    pages = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                pages.append(page_text)

    return "\n".join(pages).strip()


# ============================================================
# SKILL LIST
# ============================================================

@st.cache_data
def load_skills():
    if not SKILLS_PATH.exists():
        return []

    df = pd.read_csv(SKILLS_PATH)

    # Supports files with a Skill column or a single-column CSV.
    if df.empty:
        return []

    if "skill" in [str(c).strip().lower() for c in df.columns]:
        skill_column = next(
            c for c in df.columns
            if str(c).strip().lower() == "skill"
        )
        values = df[skill_column].dropna().astype(str).tolist()
    else:
        values = (
            df.astype(str)
            .values
            .flatten()
            .tolist()
        )

    skills = []

    for value in values:
        value = value.strip()

        if value and value.lower() not in {
            "nan", "none", "skill", "skills"
        }:
            skills.append(value)

    # Deduplicate without losing the original display spelling.
    unique = {}
    for skill in skills:
        unique.setdefault(skill.lower(), skill)

    return list(unique.values())


def extract_skills(resume_text, known_skills):
    """Find skills from the configured skills list in resume text."""
    text = clean_text(resume_text)
    found = []

    for skill in known_skills:
        normalized_skill = clean_text(skill)

        if not normalized_skill:
            continue

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(normalized_skill)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, text):
            found.append(skill)

    return sorted(set(found), key=str.lower)


# ============================================================
# RESUME CLASSIFICATION
# ============================================================

def predict_resume_category(resume_text, classifier, vectorizer):
    """Predict a resume category using the saved ML model."""
    cleaned = clean_text(resume_text)

    if not cleaned:
        return "Unable to classify", None

    features = vectorizer.transform([cleaned])
    prediction = classifier.predict(features)[0]

    confidence = None

    if hasattr(classifier, "predict_proba"):
        try:
            probabilities = classifier.predict_proba(features)[0]
            confidence = float(max(probabilities)) * 100
        except Exception:
            confidence = None

    return str(prediction), confidence


# ============================================================
# ATS SCORE
# ============================================================

def calculate_ats_score(resume_text, job_description):
    """
    Estimate keyword coverage against a job description.

    This is a simple keyword-overlap estimate, not a score from
    a commercial applicant tracking system.
    """
    resume_words = set(re.findall(r"[a-zA-Z][a-zA-Z+#.]*",
                                  clean_text(resume_text)))

    job_words = set(re.findall(r"[a-zA-Z][a-zA-Z+#.]*",
                               clean_text(job_description)))

    # Remove common words that do not help keyword matching.
    stop_words = {
        "the", "and", "for", "with", "you", "your", "are",
        "our", "this", "that", "from", "will", "have", "has",
        "into", "who", "what", "when", "where", "how", "all",
        "can", "not", "but", "any", "use", "using", "job",
        "role", "work", "years", "year", "ability", "skills",
        "skill", "experience", "required", "preferred",
    }

    job_keywords = job_words - stop_words

    if not job_keywords:
        return 0, [], []

    matched = sorted(resume_words & job_keywords)
    missing = sorted(job_keywords - resume_words)

    score = round((len(matched) / len(job_keywords)) * 100)

    return score, matched, missing


# ============================================================
# NORMALIZE CAREER DATA
# Handles common key variations from career_navigator.py.
# ============================================================

def get_value(record, *keys, default=None):
    if not isinstance(record, dict):
        return default

    normalized = {
        str(key).strip().lower().replace("_", " "): value
        for key, value in record.items()
    }

    for key in keys:
        lookup = key.strip().lower().replace("_", " ")
        if lookup in normalized:
            return normalized[lookup]

    return default


def normalize_skill_list(value):
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple, set)):
        return []

    result = []

    for item in value:
        if isinstance(item, dict):
            name = get_value(
                item, "skill", "name", "title", default=""
            )
            if name:
                result.append(str(name).strip())
        elif item is not None:
            name = str(item).strip()
            if name:
                result.append(name)

    return sorted(set(result), key=str.lower)


def normalize_graph_data(raw_graph):
    """
    Expected graph structure:
    [
        {"career": "AI Engineer",
         "skills": ["Python", "SQL", "Machine Learning"]}
    ]
    """
    careers = []

    if isinstance(raw_graph, pd.DataFrame):
        raw_graph = raw_graph.to_dict(orient="records")

    if isinstance(raw_graph, dict):
        # Allow a wrapper dictionary such as {"careers": [...]}.
        wrapped = get_value(
            raw_graph, "careers", "data", "results", default=None
        )
        if wrapped is not None:
            raw_graph = wrapped
        else:
            raw_graph = [raw_graph]

    if not isinstance(raw_graph, (list, tuple)):
        return careers

    for record in raw_graph:
        if not isinstance(record, dict):
            continue

        career_name = get_value(
            record, "career", "career_name", "name", "title"
        )

        skills = get_value(
            record,
            "skills",
            "required_skills",
            "skill_names",
            default=[],
        )

        if not career_name:
            continue

        careers.append({
            "career": str(career_name).strip(),
            "skills": normalize_skill_list(skills),
        })

    # Deduplicate career entries and merge their skills.
    merged = {}

    for item in careers:
        key = item["career"].lower()

        if key not in merged:
            merged[key] = {
                "career": item["career"],
                "skills": set(),
            }

        merged[key]["skills"].update(item["skills"])

    return [
        {
            "career": item["career"],
            "skills": sorted(item["skills"], key=str.lower),
        }
        for item in merged.values()
    ]


def normalize_recommendations(raw_recommendations):
    """Convert recommendation results into a consistent structure."""
    if isinstance(raw_recommendations, pd.DataFrame):
        raw_recommendations = raw_recommendations.to_dict(
            orient="records"
        )

    if isinstance(raw_recommendations, dict):
        wrapped = get_value(
            raw_recommendations,
            "recommendations",
            "careers",
            "results",
            default=None,
        )
        raw_recommendations = (
            wrapped if wrapped is not None
            else [raw_recommendations]
        )

    if not isinstance(raw_recommendations, (list, tuple)):
        return []

    results = []

    for record in raw_recommendations:
        if not isinstance(record, dict):
            continue

        career = get_value(
            record, "career", "career_name", "name", "title"
        )

        if not career:
            continue

        matched = normalize_skill_list(
            get_value(
                record, "matched skills", "matched", "skills matched",
                default=[]
            )
        )

        missing = normalize_skill_list(
            get_value(
                record, "missing skills", "missing",
                "skills missing", default=[]
            )
        )

        coverage = get_value(
            record, "coverage", "match percentage",
            "match_percent", "score", default=0
        )

        try:
            coverage = float(coverage)
            # Some implementations may return a fraction from 0 to 1.
            if 0 <= coverage <= 1:
                coverage *= 100
            coverage = max(0, min(100, coverage))
        except (TypeError, ValueError):
            coverage = 0

        results.append({
            "career": str(career),
            "coverage": round(coverage, 1),
            "matched": matched,
            "missing": missing,
        })

    return sorted(
        results,
        key=lambda item: item["coverage"],
        reverse=True,
    )


# ============================================================
# INTERACTIVE DONUT CHART
# ============================================================

def make_donut_chart(matched_count, missing_count, title):
    labels = ["Matched skills", "Missing skills"]
    values = [matched_count, missing_count]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                sort=False,
                marker={
                    "colors": [
                        COLORS["matched"],
                        COLORS["missing"],
                    ],
                    "line": {
                        "color": COLORS["paper"],
                        "width": 3,
                    },
                },
                textinfo="label+percent",
                textposition="outside",
                textfont={"size": 13, "color": COLORS["text"]},
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Skills: %{value}<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    total = matched_count + missing_count

    fig.update_layout(
        title={
            "text": title,
            "x": 0.03,
            "font": {"size": 18, "color": COLORS["text"]},
        },
        template=COLORS["template"],
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        font={"color": COLORS["text"]},
        height=360,
        margin={"l": 25, "r": 25, "t": 65, "b": 25},
        showlegend=True,
        legend={
            "orientation": "h",
            "y": -0.08,
            "x": 0.5,
            "xanchor": "center",
        },
        annotations=[
            {
                "text": f"<b>{total}</b><br><span>skills</span>",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {
                    "size": 22,
                    "color": COLORS["text"],
                },
            }
        ],
    )

    return fig


# ============================================================
# CAREER COMPARISON BAR CHART
# ============================================================

def make_career_bar_chart(recommendations):
    if not recommendations:
        return None

    df = pd.DataFrame(recommendations)
    df = df.sort_values("coverage", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["coverage"],
            y=df["career"],
            orientation="h",
            marker={
                "color": df["coverage"],
                "colorscale": [
                    [0, "#F87171"],
                    [0.5, "#60A5FA"],
                    [1, "#34D399"],
                ],
                "cmin": 0,
                "cmax": 100,
                "line": {"width": 0},
            },
            text=df["coverage"].map(lambda value: f"{value:.1f}%"),
            textposition="outside",
            textfont={"color": COLORS["text"], "size": 12},
            customdata=df[["matched", "missing"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Match: %{x:.1f}%<br>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title={
            "text": "Career Match Comparison",
            "x": 0.02,
            "font": {"size": 20, "color": COLORS["text"]},
        },
        template=COLORS["template"],
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        font={"color": COLORS["text"]},
        height=max(360, 58 * len(df)),
        margin={"l": 15, "r": 55, "t": 70, "b": 45},
        xaxis={
            "title": "Resume skill match (%)",
            "range": [0, 110],
            "ticksuffix": "%",
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
        },
        yaxis={
            "title": "",
            "gridcolor": COLORS["grid"],
        },
        hoverlabel={"font": {"size": 13}},
    )

    return fig


# ============================================================
# OVERALL NEO4J SKILL REQUIREMENT CHART
# ============================================================

def make_skill_demand_chart(graph_data):
    """
    Count how many career profiles require each skill.
    Counts come from get_career_graph(), not hardcoded values.
    """
    skill_counts = {}

    for career in graph_data:
        for skill in career["skills"]:
            key = skill.lower()

            if key not in skill_counts:
                skill_counts[key] = {
                    "skill": skill,
                    "count": 0,
                }

            skill_counts[key]["count"] += 1

    if not skill_counts:
        return None, pd.DataFrame()

    df = pd.DataFrame(skill_counts.values())
    df = df.sort_values(
        ["count", "skill"],
        ascending=[True, True],
    ).tail(15)

    fig = go.Figure(
        go.Bar(
            x=df["count"],
            y=df["skill"],
            orientation="h",
            marker_color=COLORS["secondary"],
            text=df["count"],
            textposition="outside",
            textfont={"color": COLORS["text"]},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Careers requiring this skill: %{x}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title={
            "text": "Most Common Skills Across Neo4j Careers",
            "x": 0.02,
            "font": {"size": 19, "color": COLORS["text"]},
        },
        template=COLORS["template"],
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        font={"color": COLORS["text"]},
        height=max(380, 30 * len(df) + 100),
        margin={"l": 15, "r": 45, "t": 70, "b": 40},
        xaxis={
            "title": "Number of career profiles",
            "dtick": 1,
            "gridcolor": COLORS["grid"],
        },
        yaxis={"title": ""},
    )

    return fig, df.sort_values("count", ascending=False)


# ============================================================
# INTERACTIVE 3D CAREER-SKILL NETWORK
# ============================================================

def make_3d_graph(graph_data, resume_skills, selected_career="All careers"):
    graph = nx.Graph()

    resume_skill_keys = {
        clean_text(skill) for skill in resume_skills
    }

    if selected_career == "All careers":
        visible_careers = graph_data
    else:
        visible_careers = [
            item for item in graph_data
            if item["career"] == selected_career
        ]

    for item in visible_careers:
        career = item["career"]
        career_id = f"career::{career}"

        graph.add_node(
            career_id,
            label=career,
            node_type="career",
            matched=False,
        )

        for skill in item["skills"]:
            skill_id = f"skill::{skill}"
            matched = clean_text(skill) in resume_skill_keys

            if not graph.has_node(skill_id):
                graph.add_node(
                    skill_id,
                    label=skill,
                    node_type="skill",
                    matched=matched,
                )
            else:
                graph.nodes[skill_id]["matched"] = (
                    graph.nodes[skill_id]["matched"] or matched
                )

            graph.add_edge(career_id, skill_id)

    if graph.number_of_nodes() == 0:
        return None

    # 3D spring layout; a fixed seed makes the graph more stable.
    positions = nx.spring_layout(
        graph,
        dim=3,
        seed=42,
        k=1.1,
        iterations=100,
    )

    edge_x, edge_y, edge_z = [], [], []

    for source, target in graph.edges():
        x0, y0, z0 = positions[source]
        x1, y1, z1 = positions[target]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    traces = [
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line={
                "color": COLORS["grid"],
                "width": 3,
            },
            hoverinfo="none",
            name="Career-skill links",
        )
    ]

    categories = [
        ("career", False, COLORS["primary"], "Careers"),
        ("skill", True, COLORS["matched"], "Matched skills"),
        ("skill", False, COLORS["secondary"], "Other skills"),
    ]

    for node_type, matched_value, color, label in categories:
        nodes = [
            node for node, attrs in graph.nodes(data=True)
            if attrs["node_type"] == node_type
            and (
                node_type == "career"
                or attrs["matched"] == matched_value
            )
        ]

        if not nodes:
            continue

        traces.append(
            go.Scatter3d(
                x=[positions[node][0] for node in nodes],
                y=[positions[node][1] for node in nodes],
                z=[positions[node][2] for node in nodes],
                mode="markers+text",
                name=label,
                text=[graph.nodes[node]["label"] for node in nodes],
                textposition="top center",
                textfont={
                    "size": 10,
                    "color": COLORS["text"],
                },
                marker={
                    "size": [
                        11 if node_type == "career" else 7
                        for _ in nodes
                    ],
                    "color": color,
                    "opacity": 0.95,
                    "line": {
                        "color": COLORS["paper"],
                        "width": 1,
                    },
                },
                customdata=[
                    [
                        graph.nodes[node]["label"],
                        graph.nodes[node]["node_type"],
                        (
                            "Yes" if graph.nodes[node]["matched"]
                            else "No"
                        ),
                    ]
                    for node in nodes
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Type: %{customdata[1]}<br>"
                    "Found in resume: %{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title={
            "text": "Interactive Career–Skill Network",
            "x": 0.02,
            "font": {"size": 20, "color": COLORS["text"]},
        },
        template=COLORS["template"],
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        font={"color": COLORS["text"]},
        height=650,
        margin={"l": 0, "r": 0, "t": 65, "b": 0},
        legend={
            "orientation": "h",
            "y": 0.02,
            "x": 0.5,
            "xanchor": "center",
        },
        scene={
            "bgcolor": COLORS["plot"],
            "xaxis": {
                "visible": False,
                "showbackground": False,
            },
            "yaxis": {
                "visible": False,
                "showbackground": False,
            },
            "zaxis": {
                "visible": False,
                "showbackground": False,
            },
            "camera": {
                "eye": {"x": 1.5, "y": 1.5, "z": 1.3}
            },
        },
    )

    return fig


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📄 AI Resume Analyzer")
st.caption(
    "Analyze your resume, compare career paths, and explore "
    "career–skill relationships from Neo4j."
)

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    st.write(
        "Upload a text-based PDF resume to start. "
        "Charts will update using the extracted skills."
    )

    show_resume_text = st.checkbox(
        "Show extracted resume text",
        value=False,
    )

    show_neo4j_analytics = st.checkbox(
        "Show Neo4j skill analytics",
        value=True,
    )

    st.divider()
    st.caption("Powered by Streamlit, Plotly, ML and Neo4j")


# ============================================================
# LOAD RESOURCES
# ============================================================

try:
    classifier, vectorizer = load_models()
except Exception as exc:
    st.error(f"Could not load the saved ML models: {exc}")
    st.info(
        "Check that model/resume_classifier.pkl and "
        "model/tfidf.pkl exist in your project."
    )
    st.stop()

known_skills = load_skills()

if not known_skills:
    st.warning(
        "No skills were loaded from skills.csv. "
        "Check the file and its column names."
    )


# ============================================================
# RESUME INPUT
# ============================================================

left, right = st.columns([1.05, 1])

with left:
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help="Use a text-based PDF for best results.",
    )

with right:
    job_description = st.text_area(
        "Paste a job description (optional)",
        height=125,
        placeholder=(
            "Paste the job requirements here to calculate "
            "resume keyword coverage..."
        ),
    )


# ============================================================
# FETCH NEO4J CAREER GRAPH
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def fetch_career_graph():
    raw_data = get_career_graph()
    return normalize_graph_data(raw_data)


try:
    with st.spinner("Loading career and skill relationships from Neo4j..."):
        neo4j_careers = fetch_career_graph()
    neo4j_error = None
except Exception as exc:
    neo4j_careers = []
    neo4j_error = str(exc)


if neo4j_error:
    st.warning(
        "The Neo4j career graph could not be loaded. "
        "Resume classification and ATS analysis can still work."
    )
    with st.expander("Neo4j connection details"):
        st.code(neo4j_error)


# ============================================================
# ANALYZE RESUME
# ============================================================

if uploaded_file is not None:
    try:
        with st.spinner("Reading your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this PDF: {exc}")
        st.stop()

    if not resume_text.strip():
        st.error(
            "No readable text was found in this PDF. "
            "It may be a scanned image. Try a text-based PDF."
        )
        st.stop()

    resume_skills = extract_skills(resume_text, known_skills)

    try:
        predicted_category, confidence = predict_resume_category(
            resume_text,
            classifier,
            vectorizer,
        )
    except Exception as exc:
        predicted_category = "Unable to classify"
        confidence = None
        st.warning(f"Resume classification failed: {exc}")

    st.divider()
    st.header("📌 Resume Overview")

    # ATS calculation is shown only when a job description exists.
    if job_description.strip():
        ats_score, ats_matched, ats_missing = calculate_ats_score(
            resume_text,
            job_description,
        )
    else:
        ats_score, ats_matched, ats_missing = None, [], []

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Detected skills",
        len(resume_skills),
    )

    metric_columns[1].metric(
        "Career paths in Neo4j",
        len(neo4j_careers),
    )

    metric_columns[2].metric(
        "Predicted category",
        predicted_category,
    )

    metric_columns[3].metric(
        "ATS keyword coverage",
        f"{ats_score}%" if ats_score is not None else "—",
        help=(
            "Keyword overlap against the pasted job description. "
            "This is an estimate, not an official ATS score."
        ),
    )

    if confidence is not None:
        st.caption(
            f"Model confidence: {confidence:.1f}% "
            "(only available if the classifier supports probabilities)"
        )

    st.subheader("🧩 Skills found in your resume")

    if resume_skills:
        st.write(" · ".join(f"`{skill}`" for skill in resume_skills))
    else:
        st.info(
            "No configured skills were detected. "
            "Check your skills.csv list or the PDF text."
        )

    if show_resume_text:
        with st.expander("View extracted resume text"):
            st.text_area(
                "Extracted text",
                value=resume_text,
                height=300,
                disabled=True,
            )

    # --------------------------------------------------------
    # RECOMMENDATIONS FROM NEO4J
    # --------------------------------------------------------

    st.divider()
    st.header("🎯 Career Recommendations")

    recommendations = []

    if neo4j_careers:
        try:
            raw_recommendations = recommend_careers(resume_skills)
            recommendations = normalize_recommendations(
                raw_recommendations
            )
        except Exception as exc:
            st.warning(
                f"Could not calculate career recommendations: {exc}"
            )

    if recommendations:
        best = recommendations[0]

        a, b, c = st.columns(3)
        a.metric("Best match", best["career"])
        b.metric("Match percentage", f'{best["coverage"]:.1f}%')
        c.metric("Skills detected", len(resume_skills))

        # Interactive career comparison chart.
        bar_chart = make_career_bar_chart(recommendations)

        if bar_chart is not None:
            st.plotly_chart(
                bar_chart,
                use_container_width=True,
                key="career_comparison_bar",
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

        # Choose a career to explore its skill requirements.
        career_names = [item["career"] for item in recommendations]

        selected_career = st.selectbox(
            "Explore a career",
            options=career_names,
            index=0,
            key="selected_career_details",
        )

        selected_data = next(
            (
                item for item in recommendations
                if item["career"] == selected_career
            ),
            recommendations[0],
        )

        # Donut chart for matched vs missing skills.
        donut_col, detail_col = st.columns([1, 1.2])

        with donut_col:
            st.plotly_chart(
                make_donut_chart(
                    len(selected_data["matched"]),
                    len(selected_data["missing"]),
                    f'{selected_data["career"]}: Skill Coverage',
                ),
                use_container_width=True,
                key="resume_skill_donut",
                config={"displaylogo": False},
            )

        with detail_col:
            st.subheader(selected_data["career"])
            st.progress(
                int(selected_data["coverage"]),
                text=f'Match: {selected_data["coverage"]:.1f}%',
            )

            st.markdown("**✅ Skills matched**")
            if selected_data["matched"]:
                for skill in selected_data["matched"]:
                    st.markdown(f"- {skill}")
            else:
                st.write("No matched skills were returned.")

            st.markdown("**📚 Skills to learn**")
            if selected_data["missing"]:
                for skill in selected_data["missing"]:
                    st.markdown(f"- {skill}")
            else:
                st.success(
                    "No missing skills were returned for this career."
                )

        with st.expander("View all career recommendations"):
            for item in recommendations:
                st.markdown(
                    f"### {item['career']} — {item['coverage']:.1f}%"
                )

                st.write(
                    "**Matched:** "
                    + (
                        ", ".join(item["matched"])
                        if item["matched"] else "None returned"
                    )
                )

                st.write(
                    "**Missing:** "
                    + (
                        ", ".join(item["missing"])
                        if item["missing"] else "None returned"
                    )
                )

                st.divider()

    elif neo4j_careers:
        st.info(
            "Neo4j career data is available, but no usable "
            "recommendation records were returned. Check the "
            "output structure of recommend_careers() in "
            "career_navigator.py."
        )
    else:
        st.info(
            "Career recommendations require a working Neo4j "
            "connection and career–skill relationships."
        )

    # --------------------------------------------------------
    # ATS KEYWORD BREAKDOWN
    # --------------------------------------------------------

    if job_description.strip():
        st.divider()
        st.header("📝 Job Description Keyword Analysis")

        ats_col1, ats_col2 = st.columns(2)

        with ats_col1:
            st.metric("Keyword coverage", f"{ats_score}%")
            st.markdown("**Keywords found in resume**")

            if ats_matched:
                st.write(", ".join(ats_matched))
            else:
                st.write("No matching keywords detected.")

        with ats_col2:
            st.markdown("**Keywords not found**")

            if ats_missing:
                st.write(", ".join(ats_missing))
            else:
                st.success("All extracted job keywords were found.")

        st.caption(
            "This estimate measures keyword overlap only. "
            "It does not assess formatting, recruiter preferences, "
            "or the behavior of a specific ATS product."
        )

else:
    st.info(
        "👆 Upload a PDF resume to see your skill analysis, "
        "career comparisons, and resume-specific donut chart."
    )


# ============================================================
# NEO4J CAREER-SKILL ANALYTICS
# ============================================================

st.divider()
st.header("🕸️ Neo4j Career & Skill Analytics")

if neo4j_careers:
    career_count = len(neo4j_careers)

    all_skills = sorted(
        {
            skill
            for career in neo4j_careers
            for skill in career["skills"]
        },
        key=str.lower,
    )

    relationship_count = sum(
        len(career["skills"]) for career in neo4j_careers
    )

    graph_m1, graph_m2, graph_m3 = st.columns(3)

    graph_m1.metric("Career profiles", career_count)
    graph_m2.metric("Unique skills", len(all_skills))
    graph_m3.metric("Career-skill links", relationship_count)

    # Overall Neo4j skill demand.
    if show_neo4j_analytics:
        skill_chart, skill_frequency = make_skill_demand_chart(
            neo4j_careers
        )

        if skill_chart is not None:
            st.plotly_chart(
                skill_chart,
                use_container_width=True,
                key="neo4j_skill_frequency",
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )

        with st.expander("View career-skill statistics"):
            if not skill_frequency.empty:
                st.dataframe(
                    skill_frequency.rename(
                        columns={
                            "skill": "Skill",
                            "count": "Number of careers requiring skill",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # Keep the interactive 3D graph.
    graph_options = ["All careers"] + [
        item["career"] for item in neo4j_careers
    ]

    graph_selection = st.selectbox(
        "Filter the 3D network",
        options=graph_options,
        index=0,
        key="network_career_filter",
    )

    # Uses resume skills if uploaded, otherwise shows all skills
    # in the neutral/unmatched color scheme.
    graph_resume_skills = (
        resume_skills if uploaded_file is not None else []
    )

    try:
        network_figure = make_3d_graph(
            neo4j_careers,
            graph_resume_skills,
            graph_selection,
        )

        if network_figure is not None:
            st.plotly_chart(
                network_figure,
                use_container_width=True,
                key="neo4j_3d_network",
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True,
                },
            )

            st.caption(
                "Drag to rotate · Scroll to zoom · Hover over a node "
                "to inspect its name and type. Green nodes are skills "
                "detected in the uploaded resume."
            )
    except Exception as exc:
        st.error(f"Could not render the 3D network: {exc}")

else:
    st.warning(
        "The Neo4j career–skill graph is empty or unavailable. "
        "Verify your .env connection settings and the "
        "get_career_graph() query in career_navigator.py."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "AI Resume Analyzer · ML classification · ATS keyword estimate "
    "· Neo4j career knowledge graph · Interactive Plotly analytics"
)
