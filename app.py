"""
RAG System — Streamlit Interactive Dashboard
=============================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import time
import os
import json

from config import (
    CHUNK_SIZES, EMBEDDING_MODELS, VECTOR_DBS, LLM_MODELS,
    TOP_K, TEST_QUERIES, NUM_DOCUMENTS, RESULTS_DIR,
)
from data_loader import load_arxiv_data
from chunking import chunk_documents
from embeddings import get_embedding_model, embed_chunks
from vector_store import create_vector_store
from retriever import retrieve
from prompt_builder import build_prompt
from generator import load_llm, generate_answer
from evaluator import compute_rouge, compute_bleu
from utils import log


# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG System — ArXiv Papers",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom Styling ──────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #a0a0c0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .chunk-box {
        background: rgba(255,255,255,0.05);
        border-left: 3px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
    }
    .answer-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ─── Cached Data Loading ──────────────────────────────────────
@st.cache_resource(show_spinner="📄 Loading research papers...")
def load_data():
    """Load and cache the dataset."""
    docs, abstracts, raw = load_arxiv_data(NUM_DOCUMENTS)
    return docs, abstracts


@st.cache_resource(show_spinner="✂️ Chunking documents...")
def get_chunks(chunk_size, _docs_tuple):
    """Chunk documents with caching. _docs_tuple is used for cache key."""
    docs = list(_docs_tuple)
    return chunk_documents(docs, chunk_size=chunk_size)


@st.cache_resource(show_spinner="🔢 Loading embedding model...")
def get_emb_model(model_key):
    """Load embedding model with caching."""
    return get_embedding_model(model_key)


@st.cache_resource(show_spinner="🧮 Generating embeddings...")
def get_embeddings(_emb_model, chunk_size, emb_key, _docs_tuple):
    """Generate embeddings with caching. Extra args for unique cache keys."""
    docs = list(_docs_tuple)
    chunks = chunk_documents(docs, chunk_size=chunk_size)
    embeddings = embed_chunks(chunks, _emb_model)
    return chunks, embeddings


@st.cache_resource(show_spinner="📊 Building vector index...")
def build_index(db_type, _embeddings, _chunks, emb_dim, chunk_size, emb_key):
    """Build vector store index with caching."""
    collection_name = f"st_{chunk_size}_{emb_key}_{db_type}"
    store = create_vector_store(db_type, dimension=emb_dim, collection_name=collection_name)
    store.add(_embeddings, _chunks)
    return store


@st.cache_resource(show_spinner="🤖 Loading LLM (this may take a minute)...")
def get_llm(llm_key):
    """Load LLM with caching."""
    return load_llm(llm_key)


# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("Select your RAG pipeline settings:")
    st.markdown("---")

    chunk_size = st.selectbox(
        "📏 Chunk Size",
        options=CHUNK_SIZES,
        index=1,  # Default: 512
        help="Size of text chunks in characters. Smaller = more precise retrieval, Larger = more context per chunk."
    )

    emb_key = st.selectbox(
        "🔢 Embedding Model",
        options=list(EMBEDDING_MODELS.keys()),
        format_func=lambda x: f"{'MiniLM (22M, general)' if x == 'minilm' else 'BGE-Small (33M, retrieval-optimized)'}",
        help="MiniLM is a general-purpose model. BGE is specifically trained for retrieval tasks."
    )

    db_type = st.selectbox(
        "🗄️ Vector Database",
        options=VECTOR_DBS,
        format_func=lambda x: f"{'FAISS (Meta, in-memory)' if x == 'faiss' else 'ChromaDB (metadata-aware)'}",
        help="FAISS is faster for pure similarity search. ChromaDB supports metadata filtering."
    )

    llm_key = st.selectbox(
        "🤖 LLM Model",
        options=list(LLM_MODELS.keys()),
        format_func=lambda x: {
            "tinyllama": "TinyLlama 1.1B (Fast, ~8s)",
            "phi2": "Phi-2 2.7B (Better, ~60s on CPU)",
        }.get(x, x),
        help="TinyLlama is fast but less capable. Phi-2 gives better answers but is slower."
    )

    st.markdown("---")
    st.markdown("### 📊 Current Config")
    st.code(f"Chunk: {chunk_size}\nEmbed: {emb_key}\nDB: {db_type}\nLLM: {llm_key}", language="yaml")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
        "Built for Gen AI Assignment<br>Semester 6 • 2026</div>",
        unsafe_allow_html=True,
    )


# ─── Main Area ────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 RAG System for Scientific Papers</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Retrieval-Augmented Generation over 100 ArXiv Research Papers</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Live Demo", "📊 Comparison Dashboard", "📄 Sample Outputs"])


# ═══════════════════════════════════════════════════════════════
# TAB 1: LIVE DEMO
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Ask a Question")
    st.markdown("Type a question about machine learning, statistics, or AI — the system will search 100 research papers and generate an answer.")

    # Sample questions dropdown
    sample_q = st.selectbox(
        "Or pick a sample question:",
        options=["(Type your own below)"] + TEST_QUERIES,
        index=0,
    )

    user_query = st.text_input(
        "📝 Your question:",
        value="" if sample_q == "(Type your own below)" else sample_q,
        placeholder="e.g., What are the advantages of additive models?",
    )

    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            # === Pipeline Execution ===
            with st.status("Running RAG pipeline...", expanded=True) as status:

                # Step 1: Load data
                st.write("📄 Loading research papers...")
                docs, abstracts = load_data()
                docs_tuple = tuple(docs)

                # Step 2: Embeddings + Chunks
                st.write(f"✂️ Chunking ({chunk_size} chars) + 🔢 Embedding ({emb_key})...")
                emb_model = get_emb_model(emb_key)
                chunks, embeddings = get_embeddings(emb_model, chunk_size, emb_key, docs_tuple)

                # Step 3: Index
                st.write(f"📊 Building {db_type.upper()} index...")
                store = build_index(db_type, embeddings, chunks, embeddings.shape[1], chunk_size, emb_key)

                # Step 4: Retrieve
                st.write("🔍 Retrieving relevant chunks...")
                retrieved = retrieve(user_query, emb_model, store, top_k=TOP_K)

                # Step 5: Generate
                st.write(f"🤖 Generating answer with {llm_key}...")
                pipe = get_llm(llm_key)
                prompt = build_prompt(user_query, retrieved, template_name="structured")

                start_time = time.time()
                answer = generate_answer(pipe, prompt)
                gen_time = time.time() - start_time

                # Step 6: Evaluate
                st.write("📈 Computing metrics...")
                top_doc_id = retrieved[0]["doc_id"] if retrieved else 0
                reference = abstracts[top_doc_id] if top_doc_id < len(abstracts) else ""
                rouge = compute_rouge(answer, reference)
                bleu = compute_bleu(answer, reference)

                status.update(label="✅ Pipeline complete!", state="complete")

            # === Display Results ===
            st.markdown("---")

            # Answer
            st.markdown("### 💡 Generated Answer")
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            # Metrics row
            st.markdown("### 📊 Evaluation Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("ROUGE-1", f"{rouge['rouge1']:.4f}")
            c2.metric("ROUGE-2", f"{rouge['rouge2']:.4f}")
            c3.metric("ROUGE-L", f"{rouge['rougeL']:.4f}")
            c4.metric("BLEU", f"{bleu:.4f}")
            c5.metric("Latency", f"{gen_time:.1f}s")

            # Retrieved chunks
            st.markdown("### 📄 Retrieved Context Chunks")
            for i, r in enumerate(retrieved):
                with st.expander(f"Chunk {i+1} — Score: {r['score']:.4f} | Doc #{r['doc_id']}", expanded=(i == 0)):
                    st.markdown(r["text"])

            # Config summary
            st.markdown("### 🔧 Configuration Used")
            st.json({
                "chunk_size": chunk_size,
                "embedding_model": EMBEDDING_MODELS[emb_key],
                "vector_db": db_type,
                "llm": LLM_MODELS[llm_key],
                "top_k": TOP_K,
                "temperature": 0.0,
                "num_documents": NUM_DOCUMENTS,
            })


# ═══════════════════════════════════════════════════════════════
# TAB 2: COMPARISON DASHBOARD
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Experimental Results Comparison")
    st.markdown("Pre-computed results from running all 24 configurations on 10 test queries.")

    results_path = os.path.join(RESULTS_DIR, "summary_results.csv")
    raw_results_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")

    if not os.path.exists(results_path):
        st.warning(
            "⚠️ No pre-computed results found. Run `python main.py --mode full` first to generate the comparison data."
        )
    else:
        df_summary = pd.read_csv(results_path)
        df_raw = pd.read_csv(raw_results_path) if os.path.exists(raw_results_path) else None

        # Clean up column names for display
        display_cols = ["chunk_size", "embedding", "vector_db", "llm",
                        "rouge1", "rouge2", "rougeL", "bleu",
                        "retrieval_relevance", "answer_relevance", "latency_seconds"]
        available_cols = [c for c in display_cols if c in df_summary.columns]

        st.markdown("#### Full Results Table")
        st.dataframe(
            df_summary[available_cols].style.format({
                "rouge1": "{:.4f}", "rouge2": "{:.4f}", "rougeL": "{:.4f}",
                "bleu": "{:.4f}", "retrieval_relevance": "{:.4f}",
                "answer_relevance": "{:.4f}", "latency_seconds": "{:.1f}",
            }).highlight_max(subset=["rougeL", "retrieval_relevance", "answer_relevance"], color="#2d6a4f")
             .highlight_min(subset=["latency_seconds"], color="#2d6a4f"),
            use_container_width=True,
            height=400,
        )

        st.markdown("---")

        # === Comparison Charts ===
        col1, col2 = st.columns(2)

        # --- Chunk Size Comparison ---
        with col1:
            st.markdown("#### 📏 Chunk Size Comparison")
            if "chunk_size" in df_summary.columns:
                chunk_df = df_summary.groupby("chunk_size")[["rougeL", "retrieval_relevance", "answer_relevance", "latency_seconds"]].mean().reset_index()
                st.bar_chart(chunk_df.set_index("chunk_size")[["rougeL", "retrieval_relevance", "answer_relevance"]])
                st.dataframe(chunk_df.style.format({
                    "rougeL": "{:.4f}", "retrieval_relevance": "{:.4f}",
                    "answer_relevance": "{:.4f}", "latency_seconds": "{:.1f}",
                }), use_container_width=True)

        # --- Embedding Model Comparison ---
        with col2:
            st.markdown("#### 🔢 Embedding Model Comparison")
            if "embedding" in df_summary.columns:
                emb_df = df_summary.groupby("embedding")[["rougeL", "retrieval_relevance", "answer_relevance", "latency_seconds"]].mean().reset_index()
                st.bar_chart(emb_df.set_index("embedding")[["rougeL", "retrieval_relevance", "answer_relevance"]])
                st.dataframe(emb_df.style.format({
                    "rougeL": "{:.4f}", "retrieval_relevance": "{:.4f}",
                    "answer_relevance": "{:.4f}", "latency_seconds": "{:.1f}",
                }), use_container_width=True)

        col3, col4 = st.columns(2)

        # --- Vector DB Comparison ---
        with col3:
            st.markdown("#### 🗄️ Vector Database Comparison")
            if "vector_db" in df_summary.columns:
                db_df = df_summary.groupby("vector_db")[["rougeL", "retrieval_relevance", "answer_relevance", "latency_seconds"]].mean().reset_index()
                st.bar_chart(db_df.set_index("vector_db")[["rougeL", "retrieval_relevance", "answer_relevance"]])
                st.dataframe(db_df.style.format({
                    "rougeL": "{:.4f}", "retrieval_relevance": "{:.4f}",
                    "answer_relevance": "{:.4f}", "latency_seconds": "{:.1f}",
                }), use_container_width=True)

        # --- LLM Comparison ---
        with col4:
            st.markdown("#### 🤖 LLM Comparison")
            if "llm" in df_summary.columns:
                llm_df = df_summary.groupby("llm")[["rougeL", "retrieval_relevance", "answer_relevance", "latency_seconds"]].mean().reset_index()
                st.bar_chart(llm_df.set_index("llm")[["rougeL", "retrieval_relevance", "answer_relevance"]])
                st.dataframe(llm_df.style.format({
                    "rougeL": "{:.4f}", "retrieval_relevance": "{:.4f}",
                    "answer_relevance": "{:.4f}", "latency_seconds": "{:.1f}",
                }), use_container_width=True)

        # --- Best Configuration ---
        st.markdown("---")
        st.markdown("#### 🏆 Best Configuration")
        if "rougeL" in df_summary.columns:
            best_idx = df_summary["rougeL"].idxmax()
            best = df_summary.iloc[best_idx]
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Chunk Size", int(best.get("chunk_size", 0)))
            bc2.metric("Embedding", best.get("embedding", ""))
            bc3.metric("Vector DB", best.get("vector_db", ""))
            bc4.metric("LLM", best.get("llm", ""))

            bm1, bm2, bm3 = st.columns(3)
            bm1.metric("ROUGE-L", f"{best.get('rougeL', 0):.4f}")
            bm2.metric("Answer Relevance", f"{best.get('answer_relevance', 0):.4f}")
            bm3.metric("Latency", f"{best.get('latency_seconds', 0):.1f}s")


# ═══════════════════════════════════════════════════════════════
# TAB 3: SAMPLE OUTPUTS
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📄 Sample RAG Outputs")
    st.markdown("Pre-generated examples showing the complete RAG pipeline in action.")

    raw_results_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    if os.path.exists(raw_results_path):
        df_raw = pd.read_csv(raw_results_path)

        # Get unique queries
        if "query" in df_raw.columns and "answer" in df_raw.columns:
            # Show best answer per query (highest rougeL)
            for q_id in range(min(5, df_raw["query_id"].nunique() if "query_id" in df_raw.columns else 3)):
                q_subset = df_raw[df_raw["query_id"] == q_id] if "query_id" in df_raw.columns else df_raw.head(5)
                if q_subset.empty:
                    continue

                # Best answer for this query
                best_row = q_subset.loc[q_subset["rougeL"].idxmax()] if "rougeL" in q_subset.columns else q_subset.iloc[0]

                with st.expander(f"❓ {best_row.get('query', 'N/A')}", expanded=(q_id == 0)):
                    st.markdown(f"**Config:** chunk={int(best_row.get('chunk_size', 0))}, "
                                f"embed={best_row.get('embedding', '')}, "
                                f"db={best_row.get('vector_db', '')}, "
                                f"llm={best_row.get('llm', '')}")
                    st.markdown(f"**Answer:** {best_row.get('answer', 'No answer')}")

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("ROUGE-L", f"{best_row.get('rougeL', 0):.4f}")
                    mc2.metric("BLEU", f"{best_row.get('bleu', 0):.4f}")
                    mc3.metric("Ans. Relevance", f"{best_row.get('answer_relevance', 0):.4f}")
                    mc4.metric("Latency", f"{best_row.get('latency_seconds', 0):.1f}s")

                    # Show comparison across configs for this query
                    if len(q_subset) > 1:
                        st.markdown("**All configs for this query:**")
                        compare_cols = ["chunk_size", "embedding", "vector_db", "llm", "rougeL", "answer_relevance", "latency_seconds"]
                        avail = [c for c in compare_cols if c in q_subset.columns]
                        st.dataframe(q_subset[avail].sort_values("rougeL", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("Result data doesn't contain query/answer columns.")
    else:
        st.warning("⚠️ No results found. Run `python main.py --mode full` first.")
