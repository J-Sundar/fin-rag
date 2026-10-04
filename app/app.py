"""
app.py

Streamlit frontend for the RBI Regulatory Intelligence Q&A System.
Implements a four-stage pipeline:
    1. Query Expansion  — Groq translates the user's question into official regulatory terms
    2. Hybrid Retrieval — BM25 (lexical) + Qdrant (semantic) candidate retrieval
    3. Reranking        — Cross-encoder contextual precision scoring
    4. Generation       — Groq produces a grounded, citation-backed answer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st

from src.pipeline import load_pipeline
from src.retrieval.query_expansion import expand_query
from src.retrieval.reranker import rerank_documents
from src.generation.llm_chain import format_context, _get_top_header
from src.utils.config import (
    GROQ_MODEL,
    EMBEDDING_MODEL_NAME,
    TOP_K,
    RETRIEVAL_CANDIDATES_K,
    RERANKER_MODEL_NAME,
)

# ─── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinRAG — RBI Regulatory Q&A",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App header ── */
.app-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 600;
    color: #f0f6fc;
    letter-spacing: -0.02em;
}
.app-title span { color: #d29922; }
.app-subtitle {
    font-size: 0.875rem;
    color: #8b949e;
    margin-top: 0.25rem;
    font-weight: 300;
}

/* ── Query input ── */
.stTextArea textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #d29922 !important;
    box-shadow: 0 0 0 3px rgba(210, 153, 34, 0.15) !important;
}

/* ── Primary button ── */
.stButton > button {
    background-color: #d29922 !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.6rem 1.75rem !important;
    letter-spacing: 0.03em !important;
    transition: background-color 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #e3b341 !important;
}

/* ── Pipeline status steps ── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.75rem;
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #8b949e;
    font-family: 'IBM Plex Mono', monospace;
}
.pipeline-step.active { border-color: #d29922; color: #d29922; }
.pipeline-step.done   { border-color: #238636; color: #3fb950; }
.step-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background-color: currentColor; flex-shrink: 0;
}

/* ── Query expansion badge ── */
.expansion-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #d29922;
    border-radius: 0 6px 6px 0;
    padding: 0.6rem 0.9rem;
    margin: 0.75rem 0 1.25rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    word-break: break-word;
}
.expansion-box .label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #d29922;
    margin-bottom: 0.3rem;
}

/* ── Answer box ── */
.answer-container {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.5rem 1.75rem;
    margin: 1.5rem 0 2rem 0;
    font-size: 0.95rem;
    line-height: 1.75;
    color: #e6edf3;
}
.answer-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 0.75rem;
}

/* ── Citation cards ── */
.citations-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
}
.citation-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.citation-card:hover { border-color: #30363d; }
.citation-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
}
.badge-source {
    background: #1f2d1f;
    border: 1px solid #238636;
    color: #3fb950;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
}
.badge-section {
    background: #1c2433;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
}
.badge-page {
    background: #2a2113;
    border: 1px solid #9e6a03;
    color: #d29922;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
}
.badge-score {
    background: #231d3a;
    border: 1px solid #8957e5;
    color: #d2a8ff;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
}
.citation-number {
    color: #d29922;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: auto;
}
.citation-text {
    font-size: 0.845rem;
    color: #8b949e;
    line-height: 1.65;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d;
}
.sidebar-section {
    margin-bottom: 1.5rem;
}
.sidebar-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #21262d;
}
.doc-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.4rem 0;
    font-size: 0.78rem;
    color: #c9d1d9;
    border-bottom: 1px solid #21262d;
}
.doc-item:last-child { border-bottom: none; }
.doc-dot { color: #3fb950; margin-top: 2px; flex-shrink: 0; }
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    font-size: 0.8rem;
    color: #8b949e;
    border-bottom: 1px solid #21262d;
}
.stat-row:last-child { border-bottom: none; }
.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    color: #d29922;
    font-weight: 600;
}
.pipeline-diagram {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    margin-top: 0.25rem;
}
.pipe-node {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 0.35rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    text-align: center;
}
.pipe-arrow {
    text-align: center;
    color: #30363d;
    font-size: 0.7rem;
    line-height: 0.8;
}
</style>
""", unsafe_allow_html=True)


# ─── Cached Resource Loading ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Caches pipeline resources across Streamlit reruns."""
    return load_pipeline()



def _clean_source_name(filename: str) -> str:
    """Converts 'rbi_master_direction_kyc.md' → 'RBI Master Direction KYC'"""
    return filename.replace(".md", "").replace("_", " ").title()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

CORPUS_DOCS = [
    "RBI Digital Lending Directions 2025",
    "RBI Master Direction on KYC",
    "RBI Payment Aggregators 2025",
]

with st.sidebar:
    st.markdown("""
    <div style="padding: 1.25rem 0 1rem 0; border-bottom: 1px solid #21262d; margin-bottom: 1.25rem;">
        <div style="font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1rem;
                    color: #f0f6fc; letter-spacing: -0.01em;">
            ⚖️ FinRAG
        </div>
        <div style="font-size: 0.75rem; color: #8b949e; margin-top: 0.2rem;">
            Regulatory Intelligence System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Corpus</div>', unsafe_allow_html=True)
    docs_html = "".join(
        f'<div class="doc-item"><span class="doc-dot">●</span>{doc}</div>'
        for doc in CORPUS_DOCS
    )
    st.markdown(docs_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">System</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-row"><span>Embedding</span><span class="stat-value">{EMBEDDING_MODEL_NAME.split('/')[-1]}</span></div>
    <div class="stat-row"><span>Vector DB</span><span class="stat-value">Qdrant</span></div>
    <div class="stat-row"><span>LLM</span><span class="stat-value">{GROQ_MODEL.split('/')[-1]}</span></div>
    <div class="stat-row"><span>Reranker</span><span class="stat-value">{RERANKER_MODEL_NAME.split('/')[-1]}</span></div>
    <div class="stat-row"><span>Retrieval</span><span class="stat-value">Hybrid + Rerank</span></div>
    <div class="stat-row"><span>Top-K</span><span class="stat-value">{TOP_K} (from {RETRIEVAL_CANDIDATES_K})</span></div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-diagram">
        <div class="pipe-node">Query Expansion</div>
        <div class="pipe-arrow">↓</div>
        <div class="pipe-node">BM25  +  Qdrant</div>
        <div class="pipe-arrow">↓</div>
        <div class="pipe-node">Cross-Encoder Reranker</div>
        <div class="pipe-arrow">↓</div>
        <div class="pipe-node">Groq Generation</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main UI ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div class="app-title">RBI Regulatory <span>Intelligence Q&A</span></div>
    <div class="app-subtitle">
        Ask questions over RBI Master Directions and Guidelines.
        Every answer is grounded in retrieved source circular chunks with page citations.
    </div>
</div>
""", unsafe_allow_html=True)

# Load resources (runs once, cached thereafter)
with st.spinner("Loading models and vector database..."):
    ensemble_retriever, llm_chain = get_pipeline()

# ── Query Input ──
query = st.text_area(
    label="Your question",
    placeholder="e.g. What are the KYC requirements for small payment accounts?",
    height=100,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 6])
with col1:
    submitted = st.button("Ask →", use_container_width=True)

# ── Pipeline Execution ──
if submitted and query.strip():
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stage 1: Query Expansion ──
    step1 = st.empty()
    step1.markdown(
        '<div class="pipeline-step active"><div class="step-dot"></div>'
        'Stage 1 — Translating query to legal keywords via Groq...</div>',
        unsafe_allow_html=True,
    )

    expanded = expand_query(query)

    step1.markdown(
        '<div class="pipeline-step done"><div class="step-dot"></div>'
        'Stage 1 — Query expanded</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="expansion-box">
        <div class="label">Optimised Search Keywords</div>
        {expanded}
    </div>
    """, unsafe_allow_html=True)

    # ── Stage 2: Hybrid Retrieval ──
    step2 = st.empty()
    step2.markdown(
        '<div class="pipeline-step active"><div class="step-dot"></div>'
        'Stage 2 — Running BM25 + Qdrant candidate retrieval...</div>',
        unsafe_allow_html=True,
    )

    candidate_docs = ensemble_retriever.invoke(expanded)[:RETRIEVAL_CANDIDATES_K]

    step2.markdown(
        '<div class="pipeline-step done"><div class="step-dot"></div>'
        f'Stage 2 — Retrieved {len(candidate_docs)} candidates</div>',
        unsafe_allow_html=True,
    )

    # ── Stage 3: Cross-Encoder Reranking ──
    step3 = st.empty()
    step3.markdown(
        '<div class="pipeline-step active"><div class="step-dot"></div>'
        'Stage 3 — Cross-encoder reranking candidates...</div>',
        unsafe_allow_html=True,
    )

    retrieved_docs = rerank_documents(query=query, docs=candidate_docs, top_k=TOP_K)

    step3.markdown(
        '<div class="pipeline-step done"><div class="step-dot"></div>'
        f'Stage 3 — Reranked to top {len(retrieved_docs)} chunks</div>',
        unsafe_allow_html=True,
    )

    # ── Stage 4: Generation ──
    step4 = st.empty()
    step4.markdown(
        '<div class="pipeline-step active"><div class="step-dot"></div>'
        'Stage 4 — Generating grounded answer via Groq...</div>',
        unsafe_allow_html=True,
    )

    context_text = format_context(retrieved_docs)
    answer = llm_chain.invoke({"context": context_text, "question": query})

    step4.markdown(
        '<div class="pipeline-step done"><div class="step-dot"></div>'
        'Stage 4 — Answer generated</div>',
        unsafe_allow_html=True,
    )

    # ── Answer Display ──
    st.markdown(f"""
    <div class="answer-container">
        <div class="answer-label">Answer</div>
        {answer}
    </div>
    """, unsafe_allow_html=True)

    # ── Source Citations ──
    st.markdown('<div class="citations-header">Source Chunks Retrieved</div>', unsafe_allow_html=True)

    for i, doc in enumerate(retrieved_docs, 1):
        source_raw = doc.metadata.get("source", "Unknown")
        source_label = _clean_source_name(source_raw)
        section = _get_top_header(doc.metadata)
        page = doc.metadata.get("page")
        score = doc.metadata.get("rerank_score")

        section_badge = (
            f'<span class="badge-section">{section}</span>' if section else ""
        )
        page_badge = (
            f'<span class="badge-page">p. {page}</span>' if page else ""
        )
        score_badge = (
            f'<span class="badge-score">Score: {score:+.2f}</span>' if score is not None else ""
        )

        st.markdown(f"""
        <div class="citation-card">
            <div class="citation-meta">
                <span class="badge-source">{source_label}</span>
                {page_badge}
                {section_badge}
                {score_badge}
                <span class="citation-number">#{i}</span>
            </div>
            <div class="citation-text">{doc.page_content[:500]}{"..." if len(doc.page_content) > 500 else ""}</div>
        </div>
        """, unsafe_allow_html=True)

elif submitted and not query.strip():
    st.warning("Please enter a question before submitting.")