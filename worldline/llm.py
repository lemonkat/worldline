"""
[Phase Zero: supporting infrastructure]

Large Language Model (LLM) and Embedding integrations.

This module encapsulates all DSPy model configurations, API interactions,
and local embedding initializations. It ensures that core data structures 
remain pure and do not have direct dependencies on external network calls 
or AI models.
"""

from __future__ import annotations

import os
import typing
import functools
import requests

import numpy as np
    
import dotenv
import dspy
import nltk

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
LOCAL_EMB_MODEL = None
_EMB_CACHE: dict[str, np.ndarray] = {}

TEXT_MODELS: dict[str, dspy.LM] = {}

importance: typing.Optional[dspy.Predict] = None

def init_models() -> None:
    """
    Initializes the Worldline language models by loading environment variables.

    This function is idempotent; it safely returns if already initialized. 
    It loads API keys from a `.env` file and initializes the `TEXT_MODELS` 
    dictionary with supported DSPy LM modules (Gemini, OpenAI, Anthropic, and 
    local Ollama models via API polling). It also configures a DSPy Predict 
    module for evaluating text importance. Note that the embedding model 
    is initialized separately and lazily in `get_emb`.
    """

    # load up models
    dotenv.load_dotenv()

    gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_api_key and "gemini-2.5-flash-lite" not in TEXT_MODELS:
        TEXT_MODELS["gemini-2.5-flash-lite"] = dspy.LM("gemini/gemini-2.5-flash-lite", api_key=gemini_api_key)
        TEXT_MODELS["gemini-3.6-flash"] = dspy.LM("gemini/gemini-3.6-flash", api_key=gemini_api_key)
        TEXT_MODELS["gemini-3.1-pro-preview"] = dspy.LM("gemini/gemini-3.1-pro-preview", api_key=gemini_api_key)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key and "gpt-5-nano" not in TEXT_MODELS:
        TEXT_MODELS["gpt-5-nano"] = dspy.LM("openai/gpt-5-nano", api_key=openai_api_key)
        TEXT_MODELS["gpt-5.6-terra"] = dspy.LM("openai/gpt-5.6-terra", api_key=openai_api_key)
        TEXT_MODELS["gpt-5.6-sol"] = dspy.LM("openai/gpt-5.6-sol", api_key=openai_api_key)

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key and "claude-haiku-4-5" not in TEXT_MODELS:
        TEXT_MODELS["claude-haiku-4-5"] = dspy.LM("anthropic/claude-haiku-4-5", api_key=anthropic_api_key)
        TEXT_MODELS["claude-sonnet-5"] = dspy.LM("anthropic/claude-sonnet-5", api_key=anthropic_api_key)
        TEXT_MODELS["claude-fable-5"] = dspy.LM("anthropic/claude-fable-5", api_key=anthropic_api_key)

    try:
        # Ask Ollama for the list of installed models
        ollama_addr = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        response = requests.get(f"{ollama_addr}/api/tags", timeout=1.0)
        if response.status_code == 200:
            for model in response.json().get("models", []):
                TEXT_MODELS["local-" + model["name"]] = dspy.LM(
                    "ollama_chat/" + model["name"], 
                    api_base=ollama_addr,
                    api_key="",
                )
            
    except requests.exceptions.RequestException:
        pass
            

def get_emb(text: typing.Union[str, list[str]], **kwargs) -> np.ndarray[np.float32]:
    """
    Generates vector embeddings for the provided text using a local sentence-transformers model.

    Instantiates the local embedding model defined by `EMBEDDING_MODEL_NAME` on 
    the first call. It automatically handles batch processing and hardware 
    acceleration (e.g., MPS/CUDA) via PyTorch. Features a batch-aware True LRU cache.

    Args:
        text (str or list[str]): A single string or a list of strings to embed.
        **kwargs: Additional keyword arguments for future compatibility (e.g., `is_query`).

    Returns:
        np.ndarray[np.float32]: A numpy array of the computed embeddings.
    """
    global LOCAL_EMB_MODEL, _EMB_CACHE
    
    if isinstance(text, str):
        return get_emb([text], **kwargs)[0]

    uncached_indices = []
    uncached_texts = []
    results = [None] * len(text)
    
    for i, t in enumerate(text):
        if not t:
            continue
            
        if t in _EMB_CACHE:
            # True LRU: pop and re-insert to bump to the newest position
            results[i] = _EMB_CACHE[t] = _EMB_CACHE.pop(t)
        else:
            uncached_indices.append(i)
            uncached_texts.append(t)

    if uncached_texts:
        if LOCAL_EMB_MODEL is None:
            from sentence_transformers import SentenceTransformer
            LOCAL_EMB_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        new_embs = LOCAL_EMB_MODEL.encode(uncached_texts, convert_to_numpy=True)
        
        for idx, txt, emb in zip(uncached_indices, uncached_texts, new_embs):
            emb_f32 = emb.astype(np.float32)
            results[idx] = emb_f32
            _EMB_CACHE[txt] = emb_f32
            
            # Simple LRU limit to prevent RAM leaks over long sessions
            if len(_EMB_CACHE) > 10000:
                _EMB_CACHE.pop(next(iter(_EMB_CACHE)))

    # Fill empty strings with zeros using the model's exact dimensionality
    if any(r is None for r in results):
        if LOCAL_EMB_MODEL is None:
            from sentence_transformers import SentenceTransformer
            LOCAL_EMB_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        dim = LOCAL_EMB_MODEL.get_embedding_dimension()
        zeros = np.zeros(dim, dtype=np.float32)
        for i in range(len(results)):
            if results[i] is None:
                results[i] = zeros

    return np.stack(results, axis=0)

# Global flag so we only run the download once
_NLTK_INITIALIZED = False

@functools.lru_cache(maxsize=100)
def count_sentences(text: str) -> int:
    """
    Counts the number of sentences in a given text using NLTK.

    Downloads the required NLTK 'punkt' tokenizer data on the first call. 
    Results are cached using `@functools.lru_cache` for performance.

    Args:
        text (str): The text to analyze.

    Returns:
        int: The number of sentences contained in the text.
    """
    global _NLTK_INITIALIZED
    if not _NLTK_INITIALIZED:
        # Downloads the grammar rules silently
        nltk.download("punkt", quiet=True)
        _NLTK_INITIALIZED = True
        
    return len(nltk.tokenize.sent_tokenize(text))