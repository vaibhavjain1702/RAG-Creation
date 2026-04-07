"""
inspect_data.py — Explore your project's local data

Run any section:
    python inspect_data.py --what dataset       # view the 100 HuggingFace papers
    python inspect_data.py --what chunks        # view FAISS-ready chunks + embeddings
    python inspect_data.py --what faiss         # search the FAISS index live
    python inspect_data.py --what chroma        # search the ChromaDB index live
    python inspect_data.py --what results       # view your experiment results CSV
"""

import argparse
import sys
import numpy as np

# ─────────────────────────────────────────────────
# 1. HUGGINGFACE DATASET INSPECTOR
# ─────────────────────────────────────────────────
def inspect_dataset():
    print("\n" + "="*60)
    print("  HUGGINGFACE DATASET — ccdv/arxiv-summarization")
    print("="*60)

    from datasets import load_dataset
    print("\n[Loading from local cache — no internet needed]")
    dataset = load_dataset("ccdv/arxiv-summarization", split="train", trust_remote_code=True)

    print(f"\n📦 Total papers in dataset  : {len(dataset)}")
    print(f"📑 Columns available        : {dataset.column_names}")

    # Show first 5 papers
    print("\n" + "─"*60)
    print("  FIRST 5 PAPERS (raw, uncleaned)")
    print("─"*60)
    for i in range(5):
        row = dataset[i]
        article   = row["article"]
        abstract  = row["abstract"]
        print(f"\n[Paper #{i}]")
        print(f"  Article length : {len(article)} characters")
        print(f"  Abstract length: {len(abstract)} characters")
        print(f"  Article preview: {article[:300].strip()}...")
        print(f"  Abstract       : {abstract[:200].strip()}...")

    # Show ONE paper cleaned
    print("\n" + "─"*60)
    print("  PAPER #0 — BEFORE vs AFTER CLEANING")
    print("─"*60)
    from data_loader import clean_text
    raw   = dataset[0]["article"]
    clean = clean_text(raw)
    print(f"\n  Raw   (first 400 chars): {raw[:400]}")
    print(f"\n  Clean (first 400 chars): {clean[:400]}")
    print(f"\n  Raw length  : {len(raw)} chars")
    print(f"  Clean length: {len(clean)} chars")


# ─────────────────────────────────────────────────
# 2. CHUNKS + EMBEDDINGS INSPECTOR
# ─────────────────────────────────────────────────
def inspect_chunks():
    print("\n" + "="*60)
    print("  CHUNKS + EMBEDDINGS (built live from 100 papers)")
    print("="*60)

    from data_loader import load_arxiv_data
    from chunking import chunk_documents
    from embeddings import get_embedding_model, embed_chunks
    from config import NUM_DOCUMENTS

    print("\n[Step 1] Loading 100 papers...")
    documents, abstracts, _ = load_arxiv_data(NUM_DOCUMENTS)
    print(f"  ✓ {len(documents)} papers loaded")
    print(f"  ✓ Average length: {int(np.mean([len(d) for d in documents]))} chars")

    print("\n[Step 2] Chunking with size=512...")
    chunks = chunk_documents(documents, chunk_size=512)
    print(f"  ✓ {len(chunks)} chunks created")

    print("\n  First 3 chunks:")
    for c in chunks[:3]:
        print(f"\n  Chunk #{c['chunk_id']} | doc_id={c['doc_id']} | size={len(c['text'])} chars")
        print(f"  Text: {c['text'][:200]}...")

    print("\n[Step 3] Generating embeddings (MiniLM)...")
    model = get_embedding_model("minilm")
    embeddings = embed_chunks(chunks, model)
    print(f"  ✓ Embeddings shape: {embeddings.shape}  (rows=chunks, cols=384 dimensions)")
    print(f"  ✓ Data type       : {embeddings.dtype}")
    print(f"  ✓ First vector    : {embeddings[0][:8]}...  (first 8 of 384 numbers)")
    print(f"  ✓ Vector norm     : {np.linalg.norm(embeddings[0]):.4f}  (should be 1.0 — L2 normalized)")


# ─────────────────────────────────────────────────
# 3. FAISS INDEX INSPECTOR
# ─────────────────────────────────────────────────
def inspect_faiss():
    print("\n" + "="*60)
    print("  FAISS INDEX — Live Search")
    print("="*60)

    from data_loader import load_arxiv_data
    from chunking import chunk_documents
    from embeddings import get_embedding_model, embed_chunks, embed_query
    from vector_store import create_vector_store
    from config import NUM_DOCUMENTS, TOP_K

    print("\n[Building FAISS index from 100 papers with chunk_size=512, MiniLM]")
    documents, _, _ = load_arxiv_data(NUM_DOCUMENTS)
    chunks = chunk_documents(documents, chunk_size=512)
    model = get_embedding_model("minilm")
    embeddings = embed_chunks(chunks, model)

    store = create_vector_store("faiss", dimension=embeddings.shape[1])
    store.add(embeddings, chunks)

    print(f"\n  ✓ FAISS index created")
    print(f"  ✓ Total vectors stored: {store.index.ntotal}")
    print(f"  ✓ Dimension          : {store.index.d}")
    print(f"  ✓ Index type         : {type(store.index).__name__}  (exact inner-product search)")

    # Interactive search
    print("\n" + "─"*60)
    print("  LIVE SEARCH — type a query to see what FAISS retrieves")
    print("  (type 'quit' to exit)")
    print("─"*60)

    while True:
        query = input("\n🔍 Query: ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            break
        results = store.search(embed_query(query, model), top_k=TOP_K)
        print(f"\n  Top-{TOP_K} results:")
        for i, r in enumerate(results):
            print(f"\n  [{i+1}] Score: {r['score']:.4f} | doc_id={r['doc_id']} | chunk_id={r['chunk_id']}")
            print(f"       {r['text'][:250]}...")


# ─────────────────────────────────────────────────
# 4. CHROMADB INDEX INSPECTOR
# ─────────────────────────────────────────────────
def inspect_chroma():
    print("\n" + "="*60)
    print("  CHROMADB INDEX — Live Search")
    print("="*60)

    from data_loader import load_arxiv_data
    from chunking import chunk_documents
    from embeddings import get_embedding_model, embed_chunks, embed_query
    from vector_store import create_vector_store
    from config import NUM_DOCUMENTS, TOP_K

    print("\n[Building ChromaDB index from 100 papers with chunk_size=512, MiniLM]")
    documents, _, _ = load_arxiv_data(NUM_DOCUMENTS)
    chunks = chunk_documents(documents, chunk_size=512)
    model = get_embedding_model("minilm")
    embeddings = embed_chunks(chunks, model)

    store = create_vector_store("chroma", dimension=embeddings.shape[1])
    store.add(embeddings, chunks)

    print(f"\n  ✓ ChromaDB collection created (in-memory)")
    count = store.collection.count()
    print(f"  ✓ Documents in collection: {count}")
    print(f"  ✓ Collection name        : {store.collection.name}")

    # Peek at first 3 stored entries
    print("\n  Peek at first 3 stored entries:")
    peek = store.collection.peek(limit=3)
    for i in range(len(peek["ids"])):
        print(f"\n  ID       : {peek['ids'][i]}")
        print(f"  Metadata : {peek['metadatas'][i]}")
        print(f"  Document : {peek['documents'][i][:150]}...")

    # Interactive search
    print("\n" + "─"*60)
    print("  LIVE SEARCH — type a query to see what ChromaDB retrieves")
    print("  (type 'quit' to exit)")
    print("─"*60)

    while True:
        query = input("\n🔍 Query: ").strip()
        if query.lower() in ("quit", "exit", "q", ""):
            break
        results = store.search(embed_query(query, model), top_k=TOP_K)
        print(f"\n  Top-{TOP_K} results:")
        for i, r in enumerate(results):
            print(f"\n  [{i+1}] Score: {r['score']:.4f} | doc_id={r['doc_id']} | chunk_id={r['chunk_id']}")
            print(f"       {r['text'][:250]}...")


# ─────────────────────────────────────────────────
# 5. RESULTS CSV INSPECTOR
# ─────────────────────────────────────────────────
def inspect_results():
    print("\n" + "="*60)
    print("  EXPERIMENT RESULTS")
    print("="*60)

    import pandas as pd
    import os

    results_dir = os.path.join(os.path.dirname(__file__), "results")

    # Summary
    summary_path = os.path.join(results_dir, "summary_results.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        print(f"\n📊 SUMMARY (24 configurations, averaged over 10 queries)")
        print(f"   File: results/summary_results.csv\n")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(df.to_string(index=False))

        # Best config
        if "rougeL" in df.columns:
            best = df.loc[df["rougeL"].idxmax()]
            print(f"\n🏆 Best config (by ROUGE-L): {best.to_dict()}")
    else:
        print("  ❌ summary_results.csv not found. Run an experiment first.")

    # Raw
    raw_path = os.path.join(results_dir, "evaluation_results.csv")
    if os.path.exists(raw_path):
        df_raw = pd.read_csv(raw_path)
        print(f"\n\n📋 RAW DATA — first 5 rows of evaluation_results.csv")
        print(f"   Total rows: {len(df_raw)}")
        print(df_raw.head(5).to_string(index=False))
    else:
        print("  ❌ evaluation_results.csv not found.")


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect your RAG project's local data")
    parser.add_argument(
        "--what",
        choices=["dataset", "chunks", "faiss", "chroma", "results"],
        default="results",
        help=(
            "dataset  → view the 100 HuggingFace papers locally\n"
            "chunks   → view chunks and embedding vectors\n"
            "faiss    → build FAISS index and search it live\n"
            "chroma   → build ChromaDB and search it live\n"
            "results  → view your experiment CSV results"
        )
    )
    args = parser.parse_args()

    if args.what == "dataset":
        inspect_dataset()
    elif args.what == "chunks":
        inspect_chunks()
    elif args.what == "faiss":
        inspect_faiss()
    elif args.what == "chroma":
        inspect_chroma()
    elif args.what == "results":
        inspect_results()
