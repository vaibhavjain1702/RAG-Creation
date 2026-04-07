"""
Module 7: LLM-based Answer Generation.
Loads and runs open-source LLMs for answer generation from prompts.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import LLM_MODELS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P
from utils import log


# Cache loaded models
_pipeline_cache = {}


def load_llm(model_key):
    """
    Load a HuggingFace causal LM as a text-generation pipeline (cached).
    
    Args:
        model_key: Key from LLM_MODELS config (e.g., 'tinyllama', 'phi2', 'gemma2').
    
    Returns:
        transformers.Pipeline for text generation.
    """
    if model_key in _pipeline_cache:
        log(f"Using cached LLM: {model_key}")
        return _pipeline_cache[model_key]
    
    model_name = LLM_MODELS[model_key]
    log(f"Loading LLM: {model_name}...")
    
    # Determine dtype and device based on model
    # Phi-2 is numerically unstable with float16 on MPS and float32 is too large (10GB)
    # Solution: run Phi-2 on CPU with float32 (slower but stable and reliable)
    if model_key == "phi2":
        device_map = "cpu"
        torch_dtype = torch.float32
        log("  (Phi-2 running on CPU with float32 for stability)")
    elif torch.backends.mps.is_available():
        device_map = "mps"
        torch_dtype = torch.float16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    _pipeline_cache[model_key] = pipe
    log(f"Loaded {model_key} on {device_map}")
    return pipe


def generate_answer(pipe, prompt, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE, top_p=TOP_P):
    """
    Generate an answer given a prompt.
    
    Args:
        pipe: transformers text-generation pipeline.
        prompt: Full prompt string (with context and question).
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
        top_p: Nucleus sampling parameter.
    
    Returns:
        Generated answer string (only the new text, not the prompt).
    """
    output = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True if temperature > 0 else False,
        return_full_text=False,  # Only return generated text
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    
    answer = output[0]["generated_text"].strip()
    
    # Post-process: stop at common continuation patterns
    # (LLMs often keep generating additional Q&A pairs)
    stop_markers = ["\n---", "\nQuestion:", "\n\nQuestion", "---\n", "\n\n\n"]
    for marker in stop_markers:
        if marker in answer:
            answer = answer[:answer.index(marker)].strip()
    
    return answer


if __name__ == "__main__":
    # Quick test with TinyLlama (smallest model)
    pipe = load_llm("tinyllama")
    
    test_prompt = (
        "You are a helpful assistant. Answer the question based on the context.\n\n"
        "Context: Machine learning is a subset of artificial intelligence that "
        "enables systems to learn from data.\n\n"
        "Question: What is machine learning?\n\n"
        "Answer:"
    )
    
    answer = generate_answer(pipe, test_prompt)
    print(f"\nGenerated answer:\n{answer}")
