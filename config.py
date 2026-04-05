"""
Configuration file for the RAG system.
All hyperparameters and model/path settings are centralized here.
"""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

# --- Dataset ---
DATASET_NAME = "ccdv/arxiv-summarization"
NUM_DOCUMENTS = 100

# --- Chunking ---
CHUNK_SIZES = [256, 512, 1024]
CHUNK_OVERLAP_RATIO = 0.2  # 20% overlap relative to chunk size

# --- Embedding Models ---
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}

# --- Vector Databases ---
VECTOR_DBS = ["faiss", "chroma"]

# --- LLM Models ---
LLM_MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi2": "microsoft/phi-2",
}

# --- Retrieval ---
TOP_K = 5

# --- Generation ---
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
TOP_P = 0.9

# --- Evaluation ---
TEST_QUERIES = [
    "What are additive models and why are they useful in semiparametric regression?",
    "How does regularization help prevent overfitting in machine learning models?",
    "What are the main challenges of high-dimensional data in statistical estimation?",
    "How do neural networks handle non-linear relationships in data?",
    "What is the role of kernel methods in machine learning?",
    "How does gradient descent optimization work in deep learning?",
    "What are the benefits of ensemble methods over single models?",
    "How is Bayesian inference used in probabilistic modeling?",
    "What are the trade-offs between model complexity and interpretability?",
    "How do convolutional neural networks process image data?",
]
