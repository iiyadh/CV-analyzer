import re

import pandas as pd
import spacy
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from spacy.matcher import PhraseMatcher

# ---------- CONFIG ----------
W_SEMANTIC, W_TFIDF, W_SKILLS = 0.5, 0.2, 0.3

SKILLS = [
    "python", "sql", "java", "javascript", "c++", "r", "docker", "git", "linux",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "spacy",
    "nlp", "machine learning", "deep learning", "data science", "data analysis",
    "power bi", "tableau", "excel", "streamlit", "flask", "django", "fastapi",
    "aws", "azure", "gcp", "spark", "hadoop", "airflow", "mongodb", "postgresql",
    "mysql", "react", "html", "css", "statistics", "statistiques",
    "apprentissage automatique", "traitement du langage naturel",
    "analyse de données", "visualisation", "etl", "ci/cd", "agile", "scrum",
]

EDUCATION = {
    "PhD / Doctorat": r"\bph\.?d\b|doctorat|doctorate",
    "Master / Ingénieur": r"\bmaster\b|\bmsc\b|ing[ée]nieur|engineering degree|mast[èe]re",
    "Bachelor / Licence": r"\bbachelor\b|\bbsc\b|licence|\bbac\s*\+\s*3",
}

# ---------- NLP SETUP ----------
nlp = spacy.blank("xx")  # tokenizer only -> no model download
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("SKILLS", [nlp.make_doc(s) for s in SKILLS])


@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# ---------- EXTRACTION ----------
def extract_text(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(file).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(file).paragraphs)
    return ""


def extract_skills(text: str) -> set:
    doc = nlp(text)
    return {doc[s:e].text.lower() for _, s, e in matcher(doc)}


def extract_experience(text: str) -> int:
    years = re.findall(r"(\d+)\+?\s*(?:years?|yrs?|ans?)\b", text.lower())
    return max((int(y) for y in years if int(y) < 50), default=0)


def extract_education(text: str) -> str:
    for label, pattern in EDUCATION.items():
        if re.search(pattern, text.lower()):
            return label
    return "Non détecté"


# ---------- RANKING ----------
def rank(job_text: str, cvs: dict) -> pd.DataFrame:
    names, texts = list(cvs.keys()), list(cvs.values())

    # 1) semantic similarity (Sentence Transformers)
    model = load_model()
    emb = model.encode([job_text] + texts, normalize_embeddings=True)
    semantic = cosine_similarity(emb[:1], emb[1:])[0]

    # 2) TF-IDF cosine similarity
    tfidf = TfidfVectorizer(stop_words=None, ngram_range=(1, 2)).fit_transform([job_text] + texts)
    tfidf_sim = cosine_similarity(tfidf[:1], tfidf[1:])[0]

    # 3) skills match
    job_skills = extract_skills(job_text)
    rows = []
    for i, name in enumerate(names):
        cv_skills = extract_skills(texts[i])
        matched = cv_skills & job_skills
        skill_score = len(matched) / len(job_skills) if job_skills else 0.0
        final = W_SEMANTIC * semantic[i] + W_TFIDF * tfidf_sim[i] + W_SKILLS * skill_score
        rows.append({
            "Candidat": name,
            "Score final (%)": round(final * 100, 1),
            "Sémantique (%)": round(semantic[i] * 100, 1),
            "TF-IDF (%)": round(tfidf_sim[i] * 100, 1),
            "Compétences (%)": round(skill_score * 100, 1),
            "Compétences trouvées": ", ".join(sorted(matched)),
            "Compétences manquantes": ", ".join(sorted(job_skills - cv_skills)),
            "Expérience (ans)": extract_experience(texts[i]),
            "Formation": extract_education(texts[i]),
        })
    df = pd.DataFrame(rows).sort_values("Score final (%)", ascending=False)
    df.insert(0, "Rang", range(1, len(df) + 1))
    return df


# ---------- UI ----------
st.set_page_config(page_title="CV Screening Tool", layout="wide")

# ---------- SVG icons ----------
ICONS = {
    "file":      '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/>',
    "briefcase": '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    "upload":    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "users":     '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "trophy":    '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/><path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/><path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>',
    "target":    '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "activity":  '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "award":     '<circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/>',
    "check":     '<polyline points="20 6 9 17 4 12"/>',
    "x":         '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "briefcase2": '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
}

def svg(name, size=20, color="currentColor", sw=2):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:-0.2em;flex-shrink:0">{ICONS[name]}</svg>')

# ---------- Styles ----------
st.markdown("""
<style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
    }
    .hero h1 {margin: 0; font-size: 2rem; color: white; display: flex; align-items: center; gap: .6rem;}
    .hero p {margin: .3rem 0 0; opacity: .9;}
    .section-title {
        display: flex; align-items: center; gap: .5rem;
        font-size: 1.15rem; font-weight: 600; margin-bottom: .5rem;
    }
    .section-title svg {color: #7c3aed;}
    .chip {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 3px 10px 3px 8px; margin: 3px 4px 3px 0;
        border-radius: 999px; font-size: .82rem; font-weight: 500;
    }
    .chip-ok {background: #dcfce7; color: #166534;}
    .chip-ko {background: #fee2e2; color: #991b1b;}
    .sub-title {display: flex; align-items: center; gap: .4rem; font-weight: 600; margin: .8rem 0 .2rem;}
    .kpi {
        display: flex; align-items: center; gap: .9rem;
        background: rgba(124,58,237,.06); border: 1px solid rgba(124,58,237,.2);
        padding: 14px 16px; border-radius: 12px; height: 100%;
    }
    .kpi-icon {
        width: 42px; height: 42px; border-radius: 10px; flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        background: rgba(124,58,237,.14); color: #7c3aed;
    }
    .kpi-label {font-size: .78rem; opacity: .7; margin: 0;}
    .kpi-value {font-size: 1.25rem; font-weight: 700; margin: 0; line-height: 1.2;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 190px;}
    .cand-head {display: flex; align-items: center; gap: .6rem; margin-bottom: .6rem;}
    .score-badge {padding: 2px 12px; border-radius: 8px; font-weight: 600; color: white;}
    div[data-testid="stMetric"] {
        background: rgba(124,58,237,.06); border: 1px solid rgba(124,58,237,.2);
        padding: 12px 16px; border-radius: 12px;
    }
    .stButton > button[kind="primary"] {
        border-radius: 10px; padding: .6rem 1.2rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def section(icon, title):
    st.markdown(f"<div class='section-title'>{svg(icon, 22)}{title}</div>", unsafe_allow_html=True)

def kpi(icon, label, value):
    return (f"<div class='kpi'><div class='kpi-icon'>{svg(icon, 22)}</div>"
            f"<div><p class='kpi-label'>{label}</p><p class='kpi-value' title='{value}'>{value}</p></div></div>")

def chips(text, css, icon):
    items = [s.strip() for s in str(text or "").split(",") if s.strip()]
    if not items:
        return "<span style='color:#888'>—</span>"
    return "".join(f"<span class='chip {css}'>{svg(icon, 13, sw=2.5)}{s}</span>" for s in items)

def score_color(s):
    return "#16a34a" if s >= 70 else "#f59e0b" if s >= 50 else "#dc2626"

MEDAL = {1: "#f59e0b", 2: "#94a3b8", 3: "#b45309"}

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
    <h1>{svg("file", 32, "white")} CV Screening Tool</h1>
    <p>Classement de CV (FR/EN) par similarité sémantique + correspondance des compétences</p>
</div>
""", unsafe_allow_html=True)

# ---------- Inputs ----------
col_job, col_files = st.columns([3, 2], gap="large")
with col_job:
    section("briefcase", "1. Offre d'emploi")
    job = st.text_area("Offre d'emploi", height=230, label_visibility="collapsed",
                       placeholder="Collez la description du poste ici...")
with col_files:
    section("upload", "2. CV des candidats")
    files = st.file_uploader("CV (PDF ou DOCX)", type=["pdf", "docx"],
                             accept_multiple_files=True, label_visibility="collapsed")
    if files:
        st.markdown(f"<span style='color:#16a34a;font-size:.9rem'>{svg('check', 16, sw=3)} "
                    f"{len(files)} fichier(s) chargé(s)</span>", unsafe_allow_html=True)

run = st.button("Classer les candidats", type="primary", use_container_width=True)

# ---------- Processing ----------
if run:
    if not job.strip() or not files:
        st.warning("Ajoutez une offre d'emploi et au moins un CV.")
    else:
        with st.spinner("Analyse en cours..."):
            cvs = {f.name: extract_text(f) for f in files}
            cvs = {k: v for k, v in cvs.items() if v.strip()}
            df = rank(job, cvs)
        st.session_state["df"] = df

# ---------- Results ----------
df = st.session_state.get("df")
if df is not None and not df.empty:
    st.divider()

    best = df.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("users", "Candidats analysés", len(df)), unsafe_allow_html=True)
    k2.markdown(kpi("trophy", "Meilleur candidat", best["Candidat"]), unsafe_allow_html=True)
    k3.markdown(kpi("target", "Meilleur score", f"{best['Score final (%)']}%"), unsafe_allow_html=True)
    k4.markdown(kpi("activity", "Score moyen", f"{df['Score final (%)'].mean():.1f}%"), unsafe_allow_html=True)
    st.write("")

    tab1, tab2, tab3 = st.tabs(["Classement", "Détails par candidat", "Comparaison"])

    with tab1:
        pct = lambda label: st.column_config.ProgressColumn(
            label, min_value=0, max_value=100, format="%.1f%%")
        st.dataframe(
            df[["Rang", "Candidat", "Score final (%)", "Sémantique (%)",
                "TF-IDF (%)", "Compétences (%)", "Expérience (ans)", "Formation"]],
            use_container_width=True, hide_index=True,
            column_config={
                "Rang": st.column_config.NumberColumn("Rang", width="small"),
                "Score final (%)": pct("Score final"),
                "Sémantique (%)": pct("Sémantique"),
                "TF-IDF (%)": pct("TF-IDF"),
                "Compétences (%)": pct("Compétences"),
                "Expérience (ans)": st.column_config.NumberColumn("Expérience", format="%s ans"),
            },
        )
        st.download_button("Exporter en CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "classement_candidats.csv", "text/csv")

    with tab2:
        for _, r in df.iterrows():
            rank_n, score = r["Rang"], r["Score final (%)"]
            with st.expander(f"#{rank_n}  ·  {r['Candidat']}  —  {score}%", expanded=(rank_n == 1)):
                badge_icon = svg("award", 26, MEDAL[rank_n]) if rank_n in MEDAL else svg("award", 26, "#7c3aed")
                st.markdown(
                    f"<div class='cand-head'>{badge_icon}"
                    f"<b style='font-size:1.05rem'>{r['Candidat']}</b>"
                    f"<span class='score-badge' style='background:{score_color(score)}'>{score}%</span></div>",
                    unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Score final", f"{score}%")
                c2.metric("Expérience", f"{r['Expérience (ans)']} ans")
                c3.metric("Formation", r["Formation"] or "—")

                st.markdown(f"<div class='sub-title'>{svg('check', 18, '#16a34a', 3)} Compétences trouvées</div>",
                            unsafe_allow_html=True)
                st.markdown(chips(r["Compétences trouvées"], "chip-ok", "check"), unsafe_allow_html=True)
                st.markdown(f"<div class='sub-title'>{svg('x', 18, '#dc2626', 3)} Compétences manquantes</div>",
                            unsafe_allow_html=True)
                st.markdown(chips(r["Compétences manquantes"], "chip-ko", "x"), unsafe_allow_html=True)

    with tab3:
        section("activity", "Répartition des scores par candidat")
        st.bar_chart(df.set_index("Candidat")[["Sémantique (%)", "TF-IDF (%)", "Compétences (%)"]],
                     height=380)