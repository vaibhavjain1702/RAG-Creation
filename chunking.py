"""
Module 2: Text Chunking.
Splits cleaned documents into overlapping chunks using multiple strategies.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZES, CHUNK_OVERLAP_RATIO
from utils import log


def chunk_documents(documents, chunk_size, chunk_overlap=None):
    """
    Split documents into overlapping text chunks.
    
    Uses RecursiveCharacterTextSplitter which tries to split at natural
    boundaries (paragraphs -> sentences -> words) before character-level.
    
    Args:
        documents: List of cleaned text strings.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap in characters (default: chunk_size * CHUNK_OVERLAP_RATIO).
    
    Returns:
        List of dicts: [
            {
                "text": str,
                "doc_id": int,
                "chunk_id": int,
                "chunk_size_config": int
            },
            ...
        ]
    """
    if chunk_overlap is None:
        chunk_overlap = int(chunk_size * CHUNK_OVERLAP_RATIO)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    
    all_chunks = []
    global_chunk_id = 0
    
    for doc_id, doc_text in enumerate(documents):
        splits = splitter.split_text(doc_text)
        
        for split_text in splits:
            all_chunks.append({
                "text": split_text,
                "doc_id": doc_id,
                "chunk_id": global_chunk_id,
                "chunk_size_config": chunk_size,
            })
            global_chunk_id += 1
    
    log(f"Chunked {len(documents)} docs into {len(all_chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})")
    
    return all_chunks


def chunk_all_configurations(documents):
    """
    Chunk documents with all configured chunk sizes.
    
    Returns:
        Dict mapping chunk_size -> list of chunk dicts.
    """
    results = {}
    for size in CHUNK_SIZES:
        results[size] = chunk_documents(documents, chunk_size=size)
    return results


if __name__ == "__main__":
    # Quick test with sample text
    sample_docs = [
        "This is a sample document about machine learning. " * 50,
        "Neural networks are powerful models for classification. " * 40,
    ]
    
    for size in CHUNK_SIZES:
        chunks = chunk_documents(sample_docs, chunk_size=size)
        print(f"\nChunk size {size}: {len(chunks)} chunks")
        print(f"  First chunk ({len(chunks[0]['text'])} chars): "
              f"{chunks[0]['text'][:100]}...")
