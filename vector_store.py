"""
Module 4: Vector Store.
Manages FAISS and Chroma vector databases for storing and searching embeddings.
"""

import os
import numpy as np
import faiss
import chromadb
from utils import log


# ============================================================
# FAISS Vector Store
# ============================================================

class FAISSStore:
    """FAISS-based vector store for dense similarity search."""
    
    def __init__(self, dimension):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension (e.g., 384).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine with normalized vecs)
        self.chunks_metadata = []  # Parallel list of chunk metadata
        log(f"Created FAISS index (dim={dimension}, metric=Inner Product)")
    
    def add(self, embeddings, chunks):
        """
        Add embeddings and their metadata to the index.
        
        Args:
            embeddings: numpy array of shape (n, dim).
            chunks: List of chunk dicts with 'text', 'doc_id', 'chunk_id'.
        """
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.chunks_metadata.extend(chunks)
        log(f"FAISS: Added {len(chunks)} vectors (total: {self.index.ntotal})")
    
    def search(self, query_embedding, top_k=5):
        """
        Search for the most similar vectors.
        
        Args:
            query_embedding: numpy array of shape (1, dim).
            top_k: Number of results to return.
        
        Returns:
            List of dicts: [{"text", "doc_id", "chunk_id", "score"}, ...]
        """
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks_metadata):
                result = dict(self.chunks_metadata[idx])
                result["score"] = float(score)
                results.append(result)
        
        return results
    
    def save(self, path):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, path)
        log(f"FAISS index saved to {path}")
    
    def load(self, path):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(path)
        log(f"FAISS index loaded from {path}")


# ============================================================
# Chroma Vector Store
# ============================================================

class ChromaStore:
    """ChromaDB-based vector store with metadata support."""
    
    def __init__(self, collection_name="rag_chunks"):
        """
        Initialize Chroma client and collection.
        
        Args:
            collection_name: Name of the Chroma collection.
        """
        self.client = chromadb.Client()  # In-memory client
        
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        self.collection_name = collection_name
        log(f"Created Chroma collection: '{collection_name}'")
    
    def add(self, embeddings, chunks):
        """
        Add embeddings and chunks to the collection.
        
        Args:
            embeddings: numpy array of shape (n, dim).
            chunks: List of chunk dicts with 'text', 'doc_id', 'chunk_id'.
        """
        ids = [f"chunk_{c['chunk_id']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"],
                       "chunk_size_config": c["chunk_size_config"]} for c in chunks]
        
        # Chroma expects list of lists for embeddings
        emb_list = embeddings.tolist()
        
        # Add in batches (Chroma has a batch limit)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=emb_list[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
        
        log(f"Chroma: Added {len(chunks)} documents to '{self.collection_name}' "
            f"(total: {self.collection.count()})")
    
    def search(self, query_embedding, top_k=5):
        """
        Query the collection for similar documents.
        
        Args:
            query_embedding: numpy array of shape (1, dim) or (dim,).
            top_k: Number of results to return.
        
        Returns:
            List of dicts: [{"text", "doc_id", "chunk_id", "score"}, ...]
        """
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        if isinstance(query_embedding[0], (list, np.ndarray)):
            query_embedding = query_embedding[0]
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
        
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        
        results = []
        if result["documents"] and result["documents"][0]:
            for i, doc in enumerate(result["documents"][0]):
                entry = {
                    "text": doc,
                    "doc_id": result["metadatas"][0][i]["doc_id"],
                    "chunk_id": result["metadatas"][0][i]["chunk_id"],
                    "score": 1.0 - result["distances"][0][i] if result["distances"] else 0.0,
                }
                results.append(entry)
        
        return results


# ============================================================
# Factory Function
# ============================================================

def create_vector_store(db_type, dimension=384, collection_name="rag_chunks"):
    """
    Create a vector store of the specified type.
    
    Args:
        db_type: 'faiss' or 'chroma'.
        dimension: Embedding dimension.
        collection_name: Collection name (for Chroma).
    
    Returns:
        FAISSStore or ChromaStore instance.
    """
    if db_type == "faiss":
        return FAISSStore(dimension=dimension)
    elif db_type == "chroma":
        return ChromaStore(collection_name=collection_name)
    else:
        raise ValueError(f"Unknown vector store type: {db_type}")


if __name__ == "__main__":
    # Quick test with random vectors
    dim = 384
    n_vectors = 100
    
    # Generate random normalized vectors
    rng = np.random.default_rng(42)
    vectors = rng.random((n_vectors, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    chunks = [{"text": f"Chunk {i}", "doc_id": i // 10, "chunk_id": i,
               "chunk_size_config": 512} for i in range(n_vectors)]
    
    query = vectors[0:1]  # Use first vector as query
    
    for db_type in ["faiss", "chroma"]:
        print(f"\n--- {db_type.upper()} ---")
        store = create_vector_store(db_type, dimension=dim,
                                     collection_name=f"test_{db_type}")
        store.add(vectors, chunks)
        results = store.search(query, top_k=3)
        for r in results:
            print(f"  {r['text']}, score={r['score']:.4f}")
