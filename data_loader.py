"""
Module 1: Data Loading and Preprocessing.
Loads arXiv articles from HuggingFace and cleans LaTeX / formatting artifacts.
"""

import re
from datasets import load_dataset
from config import DATASET_NAME, NUM_DOCUMENTS
from utils import log


def clean_text(text):
    """
    Clean a raw arXiv article by removing LaTeX artifacts and normalizing text.
    
    Steps:
      1. Remove LaTeX environments (equation, align, figure, table, etc.)
      2. Remove inline math ($...$)
      3. Remove LaTeX commands (\\command{...})
      4. Remove citation markers (@xcite, \\cite{...})
      5. Remove reference markers (@xref, \\ref{...})
      6. Remove remaining backslash commands
      7. Normalize whitespace
    """
    # Remove LaTeX environments: \begin{...}...\end{...}
    text = re.sub(
        r"\\begin\{(equation|align|figure|table|eqnarray|displaymath|array|matrix|gather|multline)\*?\}.*?\\end\{\1\*?\}",
        " ", text, flags=re.DOTALL
    )
    
    # Remove inline math: $...$  (non-greedy)
    text = re.sub(r"\$[^$]+?\$", " ", text)
    
    # Remove display math: $$...$$
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    
    # Remove \cite{...}, \ref{...}, \label{...}
    text = re.sub(r"\\(cite|ref|label|eqref|citet|citep)\{[^}]*\}", "", text)
    
    # Remove @xcite and @xref markers
    text = re.sub(r"@xcite", "", text)
    text = re.sub(r"@xref", "", text)
    text = re.sub(r"@xmath\d*", "", text)
    
    # Remove common LaTeX commands but keep their text content
    text = re.sub(r"\\(textbf|textit|emph|text|mathrm|mathbf|mathit)\{([^}]*)\}", r"\2", text)
    
    # Remove remaining LaTeX commands (e.g., \alpha, \beta, \newcommand...)
    text = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})*", " ", text)
    
    # Remove curly braces
    text = re.sub(r"[{}]", "", text)
    
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    
    # Normalize whitespace: collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def load_arxiv_data(num_docs=NUM_DOCUMENTS):
    """
    Load arXiv articles from HuggingFace and return cleaned documents.
    
    Args:
        num_docs: Number of documents to load (default from config).
    
    Returns:
        documents: List of cleaned article text strings.
        abstracts: List of abstract strings (used as reference for evaluation).
        raw_data: The raw HuggingFace dataset subset.
    """
    log(f"Loading {num_docs} articles from '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split="train")
    raw_data = dataset.select(range(num_docs))
    
    documents = []
    abstracts = []
    
    for i, item in enumerate(raw_data):
        cleaned = clean_text(item["article"])
        documents.append(cleaned)
        abstracts.append(item.get("abstract", ""))
        
        if (i + 1) % 25 == 0:
            log(f"  Preprocessed {i + 1}/{num_docs} documents")
    
    log(f"Loaded {len(documents)} documents. "
        f"Avg length: {sum(len(d) for d in documents) // len(documents)} chars")
    
    return documents, abstracts, raw_data


if __name__ == "__main__":
    docs, abstracts, raw = load_arxiv_data(5)
    print(f"\nSample document (first 500 chars):\n{docs[0][:500]}")
    print(f"\nSample abstract:\n{abstracts[0][:300]}")
