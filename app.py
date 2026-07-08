import streamlit as st
import pandas as pd
import os
from langdetect import detect, LangDetectException

# Page Config 
st.set_page_config(
    page_title="MovieSearch — Semantic Engine",
    page_icon="🎬",
    layout="wide",
)

# Custom CSS 
# Load fonts via <link> tags (Streamlit doesn't reliably support @import inside <style>)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined';
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    font-weight: normal;
    font-style: normal;
    font-size: 18px;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    font-feature-settings: 'liga';
    vertical-align: middle;
}

/* ── Design Tokens ── */
:root {
    --bg-background: #f7f9fb;
    --bg-surface: #f7f9fb;
    --bg-surface-container-lowest: #ffffff;
    --bg-surface-container-low: #f2f4f6;
    --bg-surface-container: #eceef0;
    --bg-surface-container-high: #e6e8ea;
    --bg-surface-container-highest: #e0e3e5;
    --bg-tertiary-container: #0b1c30;

    --text-on-background: #191c1e;
    --text-on-surface: #191c1e;
    --text-on-surface-variant: #45464d;
    --text-on-tertiary: #ffffff;
    --text-on-tertiary-container: #75859d;
    --text-on-secondary: #ffffff;
    --text-on-secondary-fixed: #001a42;

    --color-primary: #000000;
    --color-secondary: #0058be;
    --color-secondary-container: #2170e4;
    --color-secondary-fixed: #d8e2ff;
    --color-secondary-fixed-dim: #adc6ff;
    --color-tertiary: #000000;

    --color-outline: #76777d;
    --color-outline-variant: #c6c6cd;

    --font-inter: 'Inter', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: var(--font-inter) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Force light background on main area */
.stApp {
    background-color: var(--bg-background) !important;
}
.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-tertiary-container) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-on-tertiary-container) !important;
}

/* Sidebar collapse button (when open, on dark background) */
section[data-testid="stSidebar"] button, 
section[data-testid="stSidebar"] button svg, 
section[data-testid="stSidebar"] button span,
section[data-testid="stSidebar"] button svg * {
    color: var(--text-on-tertiary) !important;
    fill: var(--text-on-tertiary) !important;
    stroke: var(--text-on-tertiary) !important;
}

/* Header & Sidebar expand/running controls (when collapsed or running, on light background) */
header, [data-testid="stHeader"] {
    background-color: transparent !important;
    background: transparent !important;
}
header *, 
[data-testid="stHeader"] *,
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="collapsedSidebarCodegen"] * {
    color: var(--bg-tertiary-container) !important;
    fill: var(--bg-tertiary-container) !important;
    stroke: var(--bg-tertiary-container) !important;
}
header button:hover,
[data-testid="stHeader"] button:hover {
    background-color: rgba(11, 28, 48, 0.1) !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-on-tertiary) !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: var(--text-on-tertiary) !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    color: var(--text-on-tertiary-container) !important;
}
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label:has(input:checked) {
    color: var(--text-on-tertiary) !important;
    font-weight: 600;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(117, 133, 157, 0.2) !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--color-secondary) !important;
    color: white !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    color: var(--text-on-tertiary-container) !important;
}
section[data-testid="stSidebar"] .stCheckbox input:checked + div {
    background-color: var(--color-secondary) !important;
    border-color: var(--color-secondary) !important;
}

/* ── Sidebar custom HTML ── */
.sidebar-title {
    font-family: var(--font-inter);
    font-size: 48px;
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-weight: 700;
    color: var(--text-on-tertiary);
    margin-bottom: 4px;
}
.sidebar-subtitle {
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1;
    letter-spacing: 0.05em;
    font-weight: 500;
    color: var(--text-on-tertiary-container);
    text-transform: uppercase;
    margin-bottom: 24px;
}
.sidebar-section-title {
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1;
    letter-spacing: 0.05em;
    font-weight: 500;
    color: var(--text-on-tertiary-container);
    text-transform: uppercase;
    margin-bottom: 8px;
    margin-top: 8px;
}
.sidebar-meta-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
}
.sidebar-meta-label {
    font-family: var(--font-inter);
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-on-tertiary-container);
}
.sidebar-meta-value {
    font-family: var(--font-mono);
    font-size: 12px;
    letter-spacing: 0.05em;
    font-weight: 600;
    color: var(--text-on-tertiary);
}
.sidebar-meta-value-sm {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.05em;
    font-weight: 500;
    color: var(--text-on-tertiary-container);
    word-break: break-all;
}

/* ── Nav item styling ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 2px;
    color: var(--text-on-tertiary-container);
    font-family: var(--font-inter);
    font-size: 16px;
    line-height: 1.6;
    transition: background 0.2s, color 0.2s;
    text-decoration: none;
    cursor: default;
}
.nav-item.active {
    color: var(--color-secondary-fixed-dim);
    font-weight: 700;
    border-right: 2px solid var(--color-secondary-fixed-dim);
    background: rgba(0, 0, 0, 0.2);
}
.nav-item .material-symbols-outlined {
    font-size: 20px;
}

/* ── Top NavBar ── */
.top-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 0 24px;
    height: 56px;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--color-outline-variant);
    margin-bottom: 24px;
}
.top-navbar-brand {
    font-family: var(--font-inter);
    font-size: 24px;
    line-height: 1.3;
    letter-spacing: -0.01em;
    font-weight: 900;
    color: var(--color-primary);
}
.top-navbar-nav {
    display: flex;
    gap: 32px;
    height: 100%;
}
.top-navbar-link {
    display: flex;
    align-items: center;
    height: 100%;
    font-family: var(--font-inter);
    font-size: 16px;
    color: var(--text-on-surface-variant);
    text-decoration: none;
    padding-bottom: 2px;
}
.top-navbar-link.active {
    color: var(--color-secondary);
    font-weight: 700;
    border-bottom: 2px solid var(--color-secondary);
}

/* ── Search Input Override ── */
.stTextInput > div > div > input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--color-outline-variant) !important;
    border-radius: 2px !important;
    padding: 16px 16px 16px 16px !important;
    font-family: var(--font-inter) !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    color: var(--text-on-surface) !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input::placeholder {
    color: rgba(69, 70, 77, 0.5) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--color-secondary) !important;
    box-shadow: 0 0 0 1px var(--color-secondary) !important;
}

/* ── Stats Bar ── */
.stats-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-surface-container-low);
    border: 1px solid rgba(198, 198, 205, 0.5);
    border-radius: 2px;
    margin-bottom: 16px;
}
.stats-bar .stat-group {
    display: flex;
    flex-direction: column;
}
.stats-bar .stat-label {
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1;
    letter-spacing: 0.05em;
    font-weight: 500;
    color: var(--text-on-surface-variant);
    text-transform: uppercase;
}
.stats-bar .stat-value {
    font-family: var(--font-mono);
    font-size: 20px;
    line-height: 1;
    font-weight: 600;
    color: var(--color-secondary);
    margin-top: 4px;
}
.stats-bar .stat-divider {
    width: 1px;
    height: 32px;
    background: rgba(198, 198, 205, 0.3);
}

/* ── Result Card ── */
.result-card {
    width: 100%;
    background: var(--bg-surface-container-lowest);
    border: 1px solid rgba(173, 198, 255, 0.5);
    border-radius: 2px;
    padding: 16px;
    margin-bottom: 16px;
    position: relative;
    display: flex;
    flex-direction: column;
    gap: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.result-card:hover {
    border-color: var(--color-secondary);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* ── Score Badge (top-right) ── */
.score-badge {
    position: absolute;
    top: 16px;
    right: 16px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1;
    letter-spacing: 0.05em;
    font-weight: 500;
}
.score-badge.top-score {
    background: var(--color-secondary);
    color: var(--text-on-secondary);
}
.score-badge.normal-score {
    background: var(--color-secondary-fixed);
    color: var(--text-on-secondary-fixed);
}

/* ── Card Title Row ── */
.card-title-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
}
.card-rank {
    font-family: var(--font-mono);
    font-size: 20px;
    line-height: 1;
    font-weight: 600;
    color: var(--color-secondary);
}
.card-title {
    font-family: var(--font-inter);
    font-size: 18px;
    line-height: 1.4;
    font-weight: 600;
    color: var(--color-primary);
}
.card-year {
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1;
    letter-spacing: 0.05em;
    font-weight: 500;
    color: var(--text-on-surface-variant);
}

/* ── Card Meta Row ── */
.card-meta-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
    color: var(--text-on-surface-variant);
    font-family: var(--font-inter);
    font-size: 14px;
    line-height: 1.5;
}
.card-meta-item {
    display: flex;
    align-items: center;
    gap: 4px;
}
.card-meta-item .material-symbols-outlined {
    font-size: 16px;
}

/* ── Card Description ── */
.card-description {
    font-family: var(--font-inter);
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-on-surface);
    max-width: 768px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ── Platform Badges ── */
.platforms-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 4px;
}
.platforms-label {
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 1;
    letter-spacing: 0.05em;
    font-weight: 500;
    color: var(--text-on-surface-variant);
    text-transform: uppercase;
}
.platform-badge {
    display: inline-block;
    padding: 2px 8px;
    background: var(--bg-surface-container-high);
    border: 1px solid rgba(198, 198, 205, 0.3);
    border-radius: 2px;
    font-family: var(--font-inter);
    font-size: 12px;
    font-weight: 500;
    color: var(--text-on-surface);
}

/* ── Exact match indicator ── */
.exact-match {
    display: inline-block;
    background: rgba(46, 204, 113, 0.15);
    color: #2ecc71;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 2px;
    border: 1px solid rgba(46, 204, 113, 0.3);
    margin-left: 8px;
    font-family: var(--font-mono);
    letter-spacing: 0.05em;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 1rem;
    color: var(--text-on-surface-variant);
}
.empty-state .icon {
    font-size: 3rem;
    margin-bottom: 0.8rem;
    opacity: 0.5;
}
.empty-state p {
    font-family: var(--font-inter);
    font-size: 16px;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
    color: var(--text-on-surface-variant);
}
.empty-state p strong {
    color: var(--text-on-surface);
}

/* ── Section Divider ── */
.section-divider {
    height: 1px;
    background: var(--color-outline-variant);
    margin: 16px 0;
    border: none;
    opacity: 0.5;
}

/* ── Hide default Streamlit header & footer ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# Load Data & Searchers (cached)
@st.cache_resource(show_spinner="Cargando dataset y construyendo índices…")
def load_searchers():
    """Load the dataset and instantiate all three searchers once."""
    from src.search_engine import BM25Searcher, DenseSearcher, HybridSearcher

    data_path = os.path.join(os.path.dirname(__file__), "data", "movies_with_embeddings.pkl")
    df = pd.read_pickle(data_path)

    bm25 = BM25Searcher(df)
    dense = DenseSearcher(df)
    hybrid = HybridSearcher(bm25, dense)

    return df, bm25, dense, hybrid


# Load Translator (cached)
@st.cache_resource(show_spinner="Cargando modelo de traducción ES→EN…")
def load_translator():
    """Load Helsinki-NLP/opus-mt-es-en for Cross-Lingual Query Translation."""
    from transformers import pipeline
    return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")


def translate_query(query: str) -> tuple[str, bool]:
    """Detect query language; translate to English if Spanish.

    Returns
    -------
    (final_query, was_translated) : tuple[str, bool]
    """
    try:
        lang = detect(query)
    except LangDetectException:
        return query, False

    if lang == "es":
        translator = load_translator()
        result = translator(query, max_length=512)
        return result[0]["translation_text"], True
    return query, False


df, bm25_searcher, dense_searcher, hybrid_searcher = load_searchers()

# Sidebar 
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">MovieSearch</div>'
        '<div class="sidebar-subtitle">Semantic Configuration</div>',
        unsafe_allow_html=True,
    )

    algorithm = st.radio(
        "Algoritmo de búsqueda",
        options=["Híbrido", "BM25 (Sparse)", "Dense (Semántico)"],
        index=0,
        help="Híbrido combina BM25 + Dense con fusión Alpha (Min-Max) o RRF.",
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown('<div class="sidebar-section-title">PARAMETERS</div>', unsafe_allow_html=True)

    top_k = st.slider(
        "Top Results (K)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    use_exact_match = st.checkbox(
        "Coincidencia exacta",
        value=True,
        help="Solo aplica al modo Híbrido. Prioriza películas cuyo título, cast o director contenga la consulta.",
    )

    fusion_method = st.radio(
        "Método de fusión",
        options=["Alpha (Min-Max)", "RRF"],
        index=0,
        help="Alpha interpola scores normalizados. RRF usa Reciprocal Rank Fusion.",
    )

    if fusion_method == "Alpha (Min-Max)":
        alpha = st.slider(
            "Alpha (Dense vs BM25)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="0.0 = puro BM25, 1.0 = puro Dense.",
        )
        rrf_k = 60  # unused but set a default
    else:
        rrf_k = st.slider(
            "RRF k",
            min_value=1,
            max_value=100,
            value=60,
            step=1,
            help="Constante de suavizado para Reciprocal Rank Fusion.",
        )
        alpha = 0.5  # unused but set a default

    st.markdown("---")

    st.markdown(
        f'<div class="sidebar-section-title">CORPUS METADATA</div>'
        f'<div class="sidebar-meta-row"><span class="sidebar-meta-label">Películas cargadas</span>'
        f'<span class="sidebar-meta-value">{len(df):,}</span></div>'
        f'<div style="margin-top:8px;"><span class="sidebar-meta-label">Modelo Activo</span><br/>'
        f'<span class="sidebar-meta-value-sm">multi-qa-MiniLM-L6-cos-v1</span></div>',
        unsafe_allow_html=True,
    )

# Top NavBar 
algo_map = {
    "Híbrido": "Hybrid",
    "BM25 (Sparse)": "BM25 Sparse",
    "Dense (Semántico)": "Dense Semantic",
}
current_algo_label = algo_map.get(algorithm, algorithm)

st.markdown(
    '<div class="top-navbar">'
    '<span class="top-navbar-brand">MovieSearch</span>'
    '<div class="top-navbar-nav"><span class="top-navbar-link active">Explorer</span></div>'
    '</div>',
    unsafe_allow_html=True,
)

# Search Box 
query = st.text_input(
    "¿Qué película estás buscando?",
    placeholder="Ej: Películas de ciencia ficción en el espacio...",
    label_visibility="collapsed",
)

# --- Query Translation (CLIR) ---
query_final = query
was_translated = False
if query and len(query.strip()) > 2:
    query_final, was_translated = translate_query(query.strip())
    if was_translated:
        st.markdown(
            f'<div style="'
            f'display:flex;align-items:center;gap:8px;'
            f'padding:8px 16px;margin-bottom:12px;'
            f'background:rgba(33,112,228,0.08);'
            f'border:1px solid rgba(33,112,228,0.25);'
            f'border-radius:2px;'
            f'font-family:var(--font-inter);font-size:14px;'
            f'color:var(--color-secondary);'
            f'">'
            f'<span class="material-symbols-outlined" style="font-size:18px;">translate</span>'
            f'Traducción automática: Buscando <strong style="margin-left:4px;">"{query_final}"</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

# Helpers 

PLATFORM_CONFIG = {
    "netflix": "Netflix",
    "amazon": "Amazon",
    "disney": "Disney+",
    "hulu": "Hulu",
}


def render_platforms(available_on: str) -> str:
    """Return HTML badges for each streaming platform."""
    if pd.isna(available_on) or not available_on:
        return ""
    badges = []
    for key, label in PLATFORM_CONFIG.items():
        if key in available_on.lower():
            badges.append(f'<span class="platform-badge">{label}</span>')
    if not badges:
        return ""
    badges_html = "".join(badges)
    return f'<div class="platforms-row"><span class="platforms-label">PLATFORMS:</span>{badges_html}</div>'


def render_card(row: pd.Series, rank: int, score: float, method: str) -> str:
    """Build a full result card as raw HTML (only div/span — Streamlit-safe).
    IMPORTANT: No leading whitespace — Markdown treats 4+ spaces as code blocks.
    """
    title = row.get("title", "Sin título")
    director = row.get("director", "—")
    if pd.isna(director):
        director = "—"
    listed_in = row.get("listed_in", "")
    available_on = row.get("available_on", "")
    year = row.get("release_year", "")
    duration = row.get("duration", "")
    description = row.get("description", "")
    if pd.isna(description):
        description = ""

    # Score badge
    if method == "hybrid" and score == 999.0:
        score_html = '<div class="score-badge top-score"><span class="material-symbols-outlined" style="font-size: 14px; margin-right: 4px;">verified</span>EXACT MATCH</div>'
        exact_html = '<span class="exact-match"><span class="material-symbols-outlined" style="font-size: 12px; margin-right: 4px; vertical-align: middle;">track_changes</span>COINCIDENCIA EXACTA</span>'
    else:
        score_fmt = f"{score:.4f}" if score < 10 else f"{score:.2f}"
        label_name = "SCORE" if method == "hybrid" else ("BM25" if method == "bm25" else "SIM")
        badge_class = "top-score" if rank == 1 else "normal-score"
        score_html = f'<div class="score-badge {badge_class}"><span class="material-symbols-outlined" style="font-size: 14px; margin-right: 4px;">analytics</span>{label_name}: {score_fmt}</div>'
        exact_html = ""

    # Meta items with Material Symbols Outlined icons
    meta_items = []
    if director != "—":
        meta_items.append(f'<div class="card-meta-item"><span class="material-symbols-outlined">movie</span>{director}</div>')
    if duration:
        meta_items.append(f'<div class="card-meta-item"><span class="material-symbols-outlined">schedule</span>{duration}</div>')
    if listed_in and not pd.isna(listed_in):
        meta_items.append(f'<div class="card-meta-item"><span class="material-symbols-outlined">sell</span>{listed_in}</div>')
    meta_html = "".join(meta_items)

    # Description
    desc_html = ""
    if description:
        desc_html = f'<div class="card-description">{description}</div>'

    platforms_html = render_platforms(available_on)

    return (
        f'<div class="result-card">'
        f'{score_html}'
        f'<div class="card-title-row">'
        f'<span class="card-rank">#{rank}</span>'
        f'<span class="card-title">{title}</span>'
        f'<span class="card-year">{year}</span>'
        f'{exact_html}'
        f'</div>'
        f'<div class="card-meta-row">{meta_html}</div>'
        f'{desc_html}'
        f'{platforms_html}'
        f'</div>'
    )


# Execute Search & Render 

if query:
    if algorithm.startswith("BM25"):
        results = bm25_searcher.search(query_final, top_k=top_k)
    elif algorithm.startswith("Dense"):
        results = dense_searcher.search(query_final, top_k=top_k)
    else:
        results = hybrid_searcher.search(
            query_final,
            top_k=top_k,
            method='alpha' if fusion_method == 'Alpha (Min-Max)' else 'rrf',
            alpha=alpha,
            rrf_k=rrf_k,
            use_exact_match=use_exact_match,
        )

    st.markdown(
        f'<div class="stats-bar">'
        f'<div class="stat-group"><span class="stat-label">RESULTADOS</span><span class="stat-value">{len(results)}</span></div>'
        f'<div class="stat-divider"></div>'
        f'<div class="stat-group"><span class="stat-label">ALGORITMO</span><span class="stat-value">{algorithm}</span></div>'
        f'<div class="stat-divider"></div>'
        f'<div class="stat-group"><span class="stat-label">TOP K</span><span class="stat-value">{top_k}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for r in results:
        matches = df[df["title"] == r["title"]]
        if matches.empty:
            continue
        row = matches.iloc[0]
        card_html = render_card(row, r["rank"], r["score"], r["method"])
        st.markdown(card_html, unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon"><span class="material-symbols-outlined" style="font-size: 48px;">search</span></div>'
        '<div>Escribí una consulta en la barra de búsqueda para empezar. '
        'Podés buscar en <strong>español</strong> o en <strong>inglés</strong> '
        '— el motor semántico entiende ambos idiomas.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
