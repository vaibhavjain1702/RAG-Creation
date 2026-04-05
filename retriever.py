"""
Module 5: Semantic Retrieval.
Retrieves relevant document chunks for a given query using a vector store.
"""

from embeddings import embed_query
from config import TOP_K
from utils import log, truncate_text


def retrieve(query, embedding_model, vector_store, top_k=TOP_K):
    """
    Retrieve the top-k most relevant chunks for a query.
    
    Pipeline:
      1. Embed the query using the same model used for indexing.
      2. Search the vector store for nearest neighbors.
      3. Return ranked results.
    
    Args:
        query: Query string.
        embedding_model: SentenceTransformer model.
        vector_store: FAISSStore or ChromaStore instance.
        top_k: Number of results to return.
    
    Returns:
        List of dicts: [{"text", "doc_id", "chunk_id", "score"}, ...]
    """
    query_embedding = embed_query(query, embedding_model)
    results = vector_store.search(query_embedding, top_k=top_k)
    return results


def retrieve_and_display(query, embedding_model, vector_store, top_k=TOP_K):
    """Retrieve and print results in a readable format."""
    results = retrieve(query, embedding_model, vector_store, top_k)
    
    print(f"\nQuery: {query}")
    print(f"{'─' * 60}")
    
    for i, r in enumerate(results):
        print(f"\n  [{i+1}] Score: {r['score']:.4f} | Doc: {r['doc_id']}")
        print(f"      {truncate_text(r['text'], 150)}")
    
    return results


if __name__ == "__main__":
    # Integration test
    from data_loader import load_arxiv_data
    from chunking import chunk_documents
    from embeddings import get_embedding_model, embed_chunks
    from vector_store import create_vector_store
    
    # Load a few documents
    docs, _, _ = load_arxiv_data(5)
    
    # Chunk
    chunks = chunk_documents(docs, chunk_size=512)
    
    # Embed
    model = get_embedding_model("minilm")
    embeddings = embed_chunks(chunks, model)
    
    # Index
    store = create_vector_store("faiss", dimension=embeddings.shape[1])
    store.add(embeddings, chunks)
    
    # Query
    retrieve_and_display(
        "What are the advantages of additive models?",
        model, store
    )
