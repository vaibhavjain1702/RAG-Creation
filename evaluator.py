"""
Module 8: Evaluation.
Computes automated metrics (ROUGE, BLEU, relevance) and formats comparison tables.
"""

import numpy as np
from collections import defaultdict

from utils import log, save_results_csv, save_json


def compute_rouge(generated, reference):
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    
    Args:
        generated: Generated answer string.
        reference: Reference answer string.
    
    Returns:
        Dict with rouge1, rouge2, rougeL scores.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        scores = scorer.score(reference, generated)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        log("rouge-score not installed, returning zeros", "WARN")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bleu(generated, reference):
    """
    Compute BLEU score.
    
    Args:
        generated: Generated answer string.
        reference: Reference answer string.
    
    Returns:
        BLEU score (float).
    """
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Tokenize
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(gen_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # Use smoothing to handle short sentences
        smoothie = SmoothingFunction().method1
        score = sentence_bleu(
            [ref_tokens], gen_tokens, smoothing_function=smoothie
        )
        return round(score, 4)
    except ImportError:
        log("nltk not installed, returning 0", "WARN")
        return 0.0


def compute_retrieval_relevance(retrieved_chunks, query, embedding_model):
    """
    Compute average cosine similarity between retrieved chunks and query.
    
    Args:
        retrieved_chunks: List of chunk dicts with 'text'.
        query: Query string.
        embedding_model: SentenceTransformer model.
    
    Returns:
        Average cosine similarity score (float).
    """
    from embeddings import embed_query, embed_chunks
    
    query_emb = embed_query(query, embedding_model)
    chunk_texts = [c["text"] for c in retrieved_chunks]
    
    if not chunk_texts:
        return 0.0
    
    chunk_embs = embed_chunks(chunk_texts, embedding_model)
    
    # Cosine similarity (already normalized)
    similarities = np.dot(chunk_embs, query_emb.T).flatten()
    return round(float(np.mean(similarities)), 4)


def compute_answer_relevance(answer, query, embedding_model):
    """
    Compute cosine similarity between the generated answer and the query.
    
    Args:
        answer: Generated answer string.
        query: Query string.
        embedding_model: SentenceTransformer model.
    
    Returns:
        Cosine similarity score (float).
    """
    from embeddings import embed_query
    
    query_emb = embed_query(query, embedding_model)
    answer_emb = embed_query(answer, embedding_model)
    
    similarity = float(np.dot(query_emb, answer_emb.T).flatten()[0])
    return round(similarity, 4)


def evaluate_single(query, generated_answer, reference_answer,
                    retrieved_chunks, embedding_model, latency=0.0):
    """
    Run all evaluation metrics for a single query-answer pair.
    
    Returns:
        Dict with all metric scores.
    """
    rouge = compute_rouge(generated_answer, reference_answer)
    bleu = compute_bleu(generated_answer, reference_answer)
    retrieval_rel = compute_retrieval_relevance(
        retrieved_chunks, query, embedding_model
    )
    answer_rel = compute_answer_relevance(
        generated_answer, query, embedding_model
    )
    
    return {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bleu": bleu,
        "retrieval_relevance": retrieval_rel,
        "answer_relevance": answer_rel,
        "latency_seconds": round(latency, 2),
    }


def aggregate_results(results_list):
    """
    Aggregate evaluation results across multiple queries.
    
    Args:
        results_list: List of evaluation dicts.
    
    Returns:
        Dict with averaged metrics.
    """
    if not results_list:
        return {}
    
    aggregated = defaultdict(list)
    for result in results_list:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                aggregated[key].append(value)
    
    return {key: round(np.mean(values), 4) for key, values in aggregated.items()}


def print_results_table(all_results):
    """
    Print a formatted comparison table of all configurations.
    
    Args:
        all_results: List of dicts, each containing config info + metrics.
    """
    if not all_results:
        print("No results to display.")
        return
    
    # Header
    headers = list(all_results[0].keys())
    
    # Calculate column widths
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in all_results))
              for h in headers}
    
    # Print header
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    separator = "-+-".join("-" * widths[h] for h in headers)
    
    print(f"\n{header_line}")
    print(separator)
    
    # Print rows
    for row in all_results:
        line = " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        print(line)
    
    print()


if __name__ == "__main__":
    # Quick test
    gen = "Additive models offer increased flexibility compared to linear models."
    ref = "Additive models provide flexibility over linear and generalized linear models."
    
    rouge = compute_rouge(gen, ref)
    bleu = compute_bleu(gen, ref)
    
    print(f"ROUGE scores: {rouge}")
    print(f"BLEU score: {bleu}")
