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
    
    # Determine dtype and device based on environment
    if torch.cuda.is_available():
        device_map = "cuda"
        torch_dtype = torch.float16
        log(f"  (Running on Google Colab / CUDA with float16)")
    elif torch.backends.mps.is_available():
        # Phi-2 is numerically unstable with float16 on MPS and float32 is too large (10GB)
        # Solution: run Phi-2 on CPU with float32 (slower but stable and reliable)
        if model_key == "phi2":
            device_map = "cpu"
            torch_dtype = torch.float32
            log("  (Phi-2 running on CPU with float32 for stability on Mac)")
        else:
            device_map = "mps"
            torch_dtype = torch.float16
    else:
        device_map = "cpu"
        torch_dtype = torch.float32
    
    from transformers import AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # We return a dictionary instead of a pipeline to bypass transformers' 
    # buggy pipeline() validation which crashes on PhiConfig in newer versions
    _pipeline_cache[model_key] = {"model": model, "tokenizer": tokenizer, "name": model_key}
    log(f"Loaded {model_key} on {device_map} (Raw API)")
    return _pipeline_cache[model_key]


def generate_answer(pipe_dict, prompt, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE, top_p=TOP_P):
    """
    Generate an answer given a prompt.
    """
    model = pipe_dict["model"]
    tokenizer = pipe_dict["tokenizer"]
    
    # Move inputs to same device as model
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Determine generation parameters
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True
    }
    
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False
        
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
        
    # Extract only the generated tokens (slice off the prompt)
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][prompt_length:]
    
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Post-process: stop at common continuation patterns
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
