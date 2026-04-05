"""
RAG System for ArXiv Scientific Papers
=======================================
Main entry point for the Retrieval-Augmented Generation system.

Usage:
    python main.py --mode quick     # Quick test (small subset, 1 config)
    python main.py --mode full      # Full experiment (all configs)
    python main.py --mode demo      # Interactive demo (single query)
"""

import argparse
import sys

from config import TEST_QUERIES, NUM_DOCUMENTS, TOP_K
from data_loader import load_arxiv_data
from chunking import chunk_documents
from embeddings import get_embedding_model, embed_chunks
from vector_store import create_vector_store
from retriever import retrieve, retrieve_and_display
from prompt_builder import build_prompt
from generator import load_llm, generate_answer
from evaluator import evaluate_single, compute_rouge, compute_bleu
from experiment import run_quick_experiment, run_full_experiment
from utils import log, print_separator, ensure_dirs


def run_demo():
    """
    Interactive demo mode: loads a single configuration and answers queries.
    Demonstrates the full RAG pipeline end-to-end.
    """
    print_separator("RAG SYSTEM DEMO")
    
    # Step 1: Load data
    log("Step 1: Loading data...")
    documents, abstracts, _ = load_arxiv_data(NUM_DOCUMENTS)
    
    # Step 2: Chunk documents
    log("Step 2: Chunking documents (size=512)...")
    chunks = chunk_documents(documents, chunk_size=512)
    
    # Step 3: Generate embeddings
    log("Step 3: Generating embeddings (MiniLM)...")
    emb_model = get_embedding_model("minilm")
    embeddings = embed_chunks(chunks, emb_model)
    
    # Step 4: Build vector index
    log("Step 4: Building FAISS index...")
    store = create_vector_store("faiss", dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    
    # Step 5: Load LLM
    log("Step 5: Loading LLM (TinyLlama)...")
    pipe = load_llm("tinyllama")
    
    print_separator("READY — Enter queries (type 'quit' to exit)")
    
    while True:
        query = input("\n📝 Your question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        
        # Retrieve
        print_separator("Retrieved Chunks")
        retrieved = retrieve_and_display(query, emb_model, store, top_k=TOP_K)
        
        # Generate
        prompt = build_prompt(query, retrieved, template_name="structured")
        print_separator("Generating Answer")
        answer = generate_answer(pipe, prompt)
        
        print(f"\n💡 Answer:\n{answer}")
        print_separator()


def run_pipeline_demo():
    """
    Non-interactive demo: runs the full pipeline on sample queries
    and displays results step-by-step.
    """
    print_separator("RAG PIPELINE DEMONSTRATION")
    
    # Load data
    log("Loading data...")
    documents, abstracts, _ = load_arxiv_data(NUM_DOCUMENTS)
    
    # Chunk
    log("Chunking documents...")
    chunks = chunk_documents(documents, chunk_size=512)
    
    # Embed
    log("Generating embeddings...")
    emb_model = get_embedding_model("minilm")
    embeddings = embed_chunks(chunks, emb_model)
    
    # Index
    log("Building index...")
    store = create_vector_store("faiss", dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    
    # Load LLM
    log("Loading LLM...")
    pipe = load_llm("tinyllama")
    
    # Run on sample queries
    sample_queries = TEST_QUERIES[:3]
    
    for i, query in enumerate(sample_queries):
        print_separator(f"Query {i+1}/{len(sample_queries)}")
        print(f"❓ {query}\n")
        
        # Retrieve
        retrieved = retrieve(query, emb_model, store, top_k=TOP_K)
        print("📄 Retrieved Chunks:")
        for j, r in enumerate(retrieved):
            print(f"  [{j+1}] Score: {r['score']:.4f} | Doc {r['doc_id']}")
            print(f"      {r['text'][:120]}...\n")
        
        # Generate
        prompt = build_prompt(query, retrieved, template_name="structured")
        answer = generate_answer(pipe, prompt)
        print(f"💡 Generated Answer:\n{answer}\n")
        
        # Evaluate
        reference = abstracts[retrieved[0]["doc_id"]] if retrieved else ""
        rouge = compute_rouge(answer, reference)
        bleu = compute_bleu(answer, reference)
        print(f"📊 Metrics: ROUGE-L={rouge['rougeL']:.4f}, BLEU={bleu:.4f}")
    
    print_separator("DEMO COMPLETE")


def main():
    parser = argparse.ArgumentParser(
        description="RAG System for ArXiv Scientific Papers"
    )
    parser.add_argument(
        "--mode", type=str, default="pipeline",
        choices=["quick", "full", "demo", "pipeline"],
        help="Execution mode: "
             "'quick' = small test, "
             "'full' = all configurations, "
             "'demo' = interactive mode, "
             "'pipeline' = non-interactive demo (default)"
    )
    
    args = parser.parse_args()
    ensure_dirs()
    
    if args.mode == "quick":
        log("Starting Quick Experiment...")
        results = run_quick_experiment()
        log(f"Done! {len(results)} results generated.")
        
    elif args.mode == "full":
        log("Starting Full Experiment (this will take a while)...")
        results = run_full_experiment()
        log(f"Done! {len(results)} results generated.")
        
    elif args.mode == "demo":
        run_demo()
        
    elif args.mode == "pipeline":
        run_pipeline_demo()


if __name__ == "__main__":
    main()