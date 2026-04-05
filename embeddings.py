"""
Module 3: Embedding Generation.
Generates dense vector embeddings using sentence-transformers models.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODELS
from utils import log, measure_time


# Cache loaded models to avoid reloading
_model_cache = {}


def get_embedding_model(model_key):
    """
    Load a SentenceTransformer model (cached).
    
    Args:
        model_key: Key from EMBEDDING_MODELS config (e.g., 'minilm', 'bge').
    
    Returns:
        SentenceTransformer model instance.
    """
    if model_key in _model_cache:
        log(f"Using cached model: {model_key}")
        return _model_cache[model_key]
    
    model_name = EMBEDDING_MODELS[model_key]
    log(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    _model_cache[model_key] = model
    log(f"Loaded {model_key} (dim={model.get_sentence_embedding_dimension()})")
    return model


def embed_chunks(chunks, model, batch_size=32):
    """
    Generate embeddings for a list of text chunks.
    
    Args:
        chunks: List of chunk dicts (must have 'text' key) or list of strings.
        model: SentenceTransformer model.
        batch_size: Encoding batch size.
    
    Returns:
        numpy array of shape (num_chunks, embedding_dim).
    """
    if isinstance(chunks[0], dict):
        texts = [c["text"] for c in chunks]
    else:
        texts = chunks
    
    log(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=batch_size,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    
    log(f"Embeddings shape: {embeddings.shape}")
    return np.array(embeddings)


def embed_query(query, model):
    """
    Embed a single query string.
    
    Args:
        query: Query string.
        model: SentenceTransformer model.
    
    Returns:
        numpy array of shape (1, embedding_dim).
    """
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )
    return np.array(embedding)


if __name__ == "__main__":
    # Quick test
    test_texts = [
        "Machine learning is a field of artificial intelligence.",
        "Neural networks can learn complex patterns from data.",
        "The weather today is sunny and warm.",
    ]
    
    for key in EMBEDDING_MODELS:
        model = get_embedding_model(key)
        embeddings = embed_chunks(test_texts, model)
        print(f"\n{key}: shape={embeddings.shape}")
        
        # Test similarity
        from numpy.linalg import norm
        sim_01 = np.dot(embeddings[0], embeddings[1])
        sim_02 = np.dot(embeddings[0], embeddings[2])
        print(f"  ML vs NN similarity: {sim_01:.4f}")
        print(f"  ML vs Weather similarity: {sim_02:.4f}")
