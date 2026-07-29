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

import numpy as np
    
import dotenv
import dspy
import nltk

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
LOCAL_EMB_MODEL = None
_EMB_CACHE: dict[str, np.ndarray] = {}

GEMINI_API_KEY: typing.Optional[str] = None

LM_LIGHT_NAME: str = "gemini/gemini-3.5-flash-lite"
LM_HEAVY_NAME: str = "gemini/gemini-3.1-pro"

LM_LIGHT: typing.Optional[dspy.LM] = None
LM_HEAVY: typing.Optional[dspy.LM] = None

importance: typing.Optional[dspy.Predict] = None

def init() -> None:
    """
    Initializes the Worldline language models by loading environment variables.

    This function is idempotent; it safely returns if already initialized. 
    It loads the `GEMINI_API_KEY` from a `.env` file and initializes the 
    lightweight and heavyweight language models from DSPy. It also configures 
    a DSPy Predict module for evaluating text importance. Note that the 
    embedding model is initialized separately and lazily in `get_emb`.

    Globals Modified:
        GEMINI_API_KEY (str): API key for Gemini.
        LM_LIGHT (dspy.LM): Fast model for simple tasks like scoring.
        LM_HEAVY (dspy.LM): Powerful model for complex tasks.
        importance (dspy.Predict): Preconfigured DSPy module for scoring text importance.
    """

    global GEMINI_API_KEY, LM_LIGHT, LM_HEAVY, importance

    if LM_LIGHT is not None:
        return

    # load up models
    dotenv.load_dotenv()
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

    # light and heavy LLMs for different tasks
    LM_LIGHT = dspy.LM(LM_LIGHT_NAME, api_key=GEMINI_API_KEY)
    LM_HEAVY = dspy.LM(LM_HEAVY_NAME, api_key=GEMINI_API_KEY)

    importance = dspy.Predict(
        dspy.Signature(
            "text: str -> importance: float",
            instructions = """
            Rate the importance or notability of the text and what it describes on a scale from 0 to 100. 
            0 is not important at all and 100 is extremely important. 
            Then divide by 100 to get a value between 0 and 1.
            """
        )
    )

def get_importance(text: typing.Union[str, list[str]]) -> typing.Union[float, list[float]]:
    """
    Evaluates the narrative importance of a given text or texts using an LLM.

    Leverages `dspy.Parallel` to concurrently evaluate batches of text, utilizing 
    the lightweight language model for rapid scoring.

    Args:
        text (str or list[str]): A single string or a list of strings to evaluate.

    Returns:
        float or list[float]: A value (or list of values) between 0.0 and 1.0 
            representing the importance of the text(s).
    """
    init()
    if isinstance(text, str):
        return importance(text=text, lm=LM_LIGHT).importance

    # Use native DSPy Parallel to handle threadpooling, and rely on 
    # DSPy's built-in exponential backoff for rate limiting.
    with dspy.context(lm=LM_LIGHT):
        parallel = dspy.Parallel(num_threads=15, disable_progress_bar=True)
        results = parallel([(importance, {"text": t}) for t in text])
        
    return [r.importance for r in results]

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
        init()
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
            init()
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