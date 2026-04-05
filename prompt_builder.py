"""
Module 6: Prompt Engineering.
Builds prompts by combining retrieved context with user queries.
Includes multiple templates with anti-hallucination safeguards.
"""


# ============================================================
# Prompt Templates
# ============================================================

PROMPT_TEMPLATES = {
    "basic": (
        "Based on the following research paper excerpts, answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    
    "structured": (
        "You are a scientific research assistant. Answer the question using ONLY "
        "the information provided in the context below. If the context does not "
        "contain enough information to answer the question, say \"I cannot answer "
        "this based on the provided context.\"\n\n"
        "Context from research papers:\n"
        "---\n{context}\n---\n\n"
        "Question: {question}\n\n"
        "Provide a concise, factual answer with specific details from the context:"
    ),
    
    "chain_of_thought": (
        "You are an expert AI research assistant analyzing scientific papers.\n\n"
        "Context from relevant research papers:\n"
        "---\n{context}\n---\n\n"
        "Question: {question}\n\n"
        "Instructions:\n"
        "1. First, identify which parts of the context are relevant to the question.\n"
        "2. Then, synthesize the information to form a complete answer.\n"
        "3. If the context is insufficient, explicitly state what is missing.\n"
        "4. Do NOT add information that is not in the context.\n\n"
        "Step-by-step reasoning and answer:"
    ),
}

DEFAULT_TEMPLATE = "structured"


def build_context(retrieved_chunks, max_context_length=2000):
    """
    Build a context string from retrieved chunks.
    
    Args:
        retrieved_chunks: List of dicts with 'text' and 'score' keys.
        max_context_length: Maximum total context length in characters.
    
    Returns:
        Formatted context string.
    """
    context_parts = []
    total_length = 0
    
    for i, chunk in enumerate(retrieved_chunks):
        chunk_text = chunk["text"].strip()
        
        # Check if adding this chunk would exceed the limit
        if total_length + len(chunk_text) > max_context_length:
            remaining = max_context_length - total_length
            if remaining > 100:  # Only add if there's meaningful space
                chunk_text = chunk_text[:remaining] + "..."
                context_parts.append(f"[Excerpt {i+1}]: {chunk_text}")
            break
        
        context_parts.append(f"[Excerpt {i+1}]: {chunk_text}")
        total_length += len(chunk_text)
    
    return "\n\n".join(context_parts)


def build_prompt(query, retrieved_chunks, template_name=DEFAULT_TEMPLATE,
                 max_context_length=2000):
    """
    Build a complete prompt from a query and retrieved chunks.
    
    Args:
        query: User question string.
        retrieved_chunks: List of retrieved chunk dicts.
        template_name: Name of the prompt template to use.
        max_context_length: Maximum context length.
    
    Returns:
        Formatted prompt string.
    """
    template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES[DEFAULT_TEMPLATE])
    context = build_context(retrieved_chunks, max_context_length)
    prompt = template.format(context=context, question=query)
    return prompt


if __name__ == "__main__":
    # Demo with sample data
    sample_chunks = [
        {"text": "Additive models provide an important family of models for "
                 "semiparametric regression. They offer increased flexibility "
                 "compared to linear models.", "score": 0.87},
        {"text": "Good estimators in additive models are less prone to the "
                 "curse of dimensionality.", "score": 0.82},
    ]
    
    query = "What are additive models and why are they useful?"
    
    for name in PROMPT_TEMPLATES:
        print(f"\n{'=' * 50}")
        print(f"Template: {name}")
        print(f"{'=' * 50}")
        prompt = build_prompt(query, sample_chunks, template_name=name)
        print(prompt)
