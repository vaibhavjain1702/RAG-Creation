"""
Experiment Orchestrator.
Runs the full comparative Grid across chunk sizes, embedding models,
vector databases, and LLMs. Produces evaluation results as CSV.
"""

import time
import itertools

from config import (
    CHUNK_SIZES, EMBEDDING_MODELS, VECTOR_DBS, LLM_MODELS,
    TOP_K, TEST_QUERIES, NUM_DOCUMENTS,
)
from data_loader import load_arxiv_data
from chunking import chunk_documents
from embeddings import get_embedding_model, embed_chunks
from vector_store import create_vector_store
from retriever import retrieve
from prompt_builder import build_prompt
from generator import load_llm, generate_answer
from evaluator import evaluate_single, aggregate_results, print_results_table
from utils import (
    log, print_separator, measure_time, save_results_csv, save_json, ensure_dirs,
)


def run_experiment(queries, documents, abstracts,
                   chunk_sizes=None, embedding_keys=None,
                   db_types=None, llm_keys=None,
                   prompt_template="structured"):
    """
    Run the full experimental grid.
    
    Args:
        queries: List of test query strings.
        documents: List of cleaned document strings.
        abstracts: List of abstract strings (used as references).
        chunk_sizes: List of chunk sizes to test. Defaults to config.
        embedding_keys: List of embedding model keys. Defaults to config.
        db_types: List of vector DB types. Defaults to config.
        llm_keys: List of LLM keys. Defaults to config.
        prompt_template: Prompt template name.
    
    Returns:
        List of result dicts (one per configuration × query).
    """
    chunk_sizes = chunk_sizes or CHUNK_SIZES
    embedding_keys = embedding_keys or list(EMBEDDING_MODELS.keys())
    db_types = db_types or VECTOR_DBS
    llm_keys = llm_keys or list(LLM_MODELS.keys())
    
    all_results = []
    config_id = 0
    
    total_configs = len(chunk_sizes) * len(embedding_keys) * len(db_types) * len(llm_keys)
    log(f"Running {total_configs} configurations × {len(queries)} queries = "
        f"{total_configs * len(queries)} evaluations")
    
    ensure_dirs()
    
    # === Outer loops: Indexing configurations ===
    for chunk_size in chunk_sizes:
        print_separator(f"Chunk Size: {chunk_size}")
        
        chunks = chunk_documents(documents, chunk_size=chunk_size)
        
        for emb_key in embedding_keys:
            print_separator(f"Embedding: {emb_key}")
            
            emb_model = get_embedding_model(emb_key)
            embeddings = embed_chunks(chunks, emb_model)
            emb_dim = embeddings.shape[1]
            
            for db_type in db_types:
                print_separator(f"Vector DB: {db_type}")
                
                collection_name = f"exp_c{chunk_size}_{emb_key}_{db_type}"
                store = create_vector_store(
                    db_type, dimension=emb_dim,
                    collection_name=collection_name
                )
                store.add(embeddings, chunks)
                
                # === Inner loop: LLM and query evaluation ===
                for llm_key in llm_keys:
                    config_id += 1
                    print_separator(
                        f"Config {config_id}/{total_configs}: "
                        f"chunk={chunk_size}, emb={emb_key}, "
                        f"db={db_type}, llm={llm_key}"
                    )
                    
                    try:
                        pipe = load_llm(llm_key)
                    except Exception as e:
                        log(f"Failed to load LLM '{llm_key}': {e}", "ERROR")
                        # Record failure but continue
                        for q_idx, query in enumerate(queries):
                            all_results.append({
                                "config_id": config_id,
                                "chunk_size": chunk_size,
                                "embedding": emb_key,
                                "vector_db": db_type,
                                "llm": llm_key,
                                "query_id": q_idx,
                                "query": query[:80],
                                "answer": "",
                                "error": str(e)[:100],
                                "rouge1": 0, "rouge2": 0, "rougeL": 0,
                                "bleu": 0, "retrieval_relevance": 0,
                                "answer_relevance": 0, "latency_seconds": 0,
                            })
                        continue
                    
                    query_results = []
                    
                    for q_idx, query in enumerate(queries):
                        log(f"  Query {q_idx+1}/{len(queries)}: {query[:60]}...")
                        
                        # Retrieve
                        retrieved = retrieve(query, emb_model, store, top_k=TOP_K)
                        
                        # Build prompt
                        prompt = build_prompt(query, retrieved, template_name=prompt_template)
                        
                        # Generate answer (with timing)
                        start_time = time.time()
                        try:
                            answer = generate_answer(pipe, prompt)
                        except Exception as e:
                            answer = f"[Generation Error: {str(e)[:100]}]"
                            log(f"  Generation error: {e}", "ERROR")
                        gen_latency = time.time() - start_time
                        
                        # Use abstract of the most relevant doc as reference
                        top_doc_id = retrieved[0]["doc_id"] if retrieved else 0
                        reference = abstracts[top_doc_id] if top_doc_id < len(abstracts) else ""
                        
                        # Evaluate
                        metrics = evaluate_single(
                            query=query,
                            generated_answer=answer,
                            reference_answer=reference,
                            retrieved_chunks=retrieved,
                            embedding_model=emb_model,
                            latency=gen_latency,
                        )
                        
                        result = {
                            "config_id": config_id,
                            "chunk_size": chunk_size,
                            "embedding": emb_key,
                            "vector_db": db_type,
                            "llm": llm_key,
                            "query_id": q_idx,
                            "query": query[:80],
                            "answer": answer[:200],
                            "error": "",
                            **metrics,
                        }
                        all_results.append(result)
                        query_results.append(metrics)
                    
                    # Print aggregated results for this config
                    avg = aggregate_results(query_results)
                    log(f"  Avg metrics: ROUGE-L={avg.get('rougeL', 0):.4f}, "
                        f"BLEU={avg.get('bleu', 0):.4f}, "
                        f"Relevance={avg.get('answer_relevance', 0):.4f}, "
                        f"Latency={avg.get('latency_seconds', 0):.1f}s")
    
    return all_results


def run_quick_experiment(queries=None, num_docs=20):
    """
    Run a quick experiment with a small subset for development.
    Uses only 1 chunk size, 1 embedding, 1 DB, 1 LLM.
    """
    queries = queries or TEST_QUERIES[:3]
    
    log("=== QUICK EXPERIMENT MODE ===")
    docs, abstracts, _ = load_arxiv_data(num_docs)
    
    results = run_experiment(
        queries=queries,
        documents=docs,
        abstracts=abstracts,
        chunk_sizes=[512],
        embedding_keys=["minilm"],
        db_types=["faiss"],
        llm_keys=["tinyllama"],
    )
    
    # Save results
    save_results_csv(results, "quick_results.csv")
    return results


def run_full_experiment():
    """
    Run the full experimental grid with all configurations.
    """
    log("=== FULL EXPERIMENT MODE ===")
    docs, abstracts, _ = load_arxiv_data(NUM_DOCUMENTS)
    
    results = run_experiment(
        queries=TEST_QUERIES,
        documents=docs,
        abstracts=abstracts,
    )
    
    # Save results
    save_results_csv(results, "evaluation_results.csv")
    
    # Also save as JSON for detailed inspection
    save_json(results, "evaluation_results.json")
    
    # Print summary table
    print_separator("SUMMARY — Aggregated by Configuration")
    
    # Group results by config_id and aggregate
    from collections import defaultdict
    config_groups = defaultdict(list)
    config_info = {}
    for r in results:
        cid = r["config_id"]
        config_groups[cid].append(r)
        if cid not in config_info:
            config_info[cid] = {
                "config_id": cid,
                "chunk_size": r["chunk_size"],
                "embedding": r["embedding"],
                "vector_db": r["vector_db"],
                "llm": r["llm"],
            }
    
    summary = []
    for cid, group in config_groups.items():
        avg = aggregate_results(group)
        row = {**config_info[cid], **avg}
        summary.append(row)
    
    print_results_table(summary)
    save_results_csv(summary, "summary_results.csv")
    
    return results


if __name__ == "__main__":
    run_quick_experiment()
