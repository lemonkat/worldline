"""
[Phase Zero: supporting infrastructure]

Large Language Model (LLM) and Embedding integrations.

This module encapsulates all DSPy model configurations and API interactions,
ensuring that core data structures remain pure and do not have direct 
dependencies on external network calls or AI models.
"""

from __future__ import annotations

import os
import typing
import functools

import numpy as np
    
import dotenv
import dspy
import nltk

GEMINI_API_KEY: typing.Optional[str] = None
LM_LIGHT: typing.Optional[dspy.LM] = None
LM_HEAVY: typing.Optional[dspy.LM] = None
EMB: typing.Optional[dspy.Embedder] = None
importance: typing.Optional[dspy.Predict] = None

def init() -> None:
    """
    Initialize the Worldline library by loading environment variables and model instances.

    This function is idempotent; it runs only once. It loads the `GEMINI_API_KEY` 
    from a `.env` file and initializes the lightweight language model, the heavyweight 
    language model, and the embedding model from DSPy. It also configures a DSPy 
    Predict module for evaluating Text importance.

    Globals Modified:
        GEMINI_API_KEY (str): API key for Gemini.
        LM_LIGHT (dspy.LM): Fast model for simple tasks like scoring.
        LM_HEAVY (dspy.LM): Powerful model for complex tasks.
        EMB (dspy.Embedder): Embedding model for vectorizing text.
        importance (dspy.Predict): Preconfigured DSPy module for scoring text importance.
    """

    global GEMINI_API_KEY, LM_LIGHT, LM_HEAVY, EMB, importance

    if LM_LIGHT is not None:
        return

    # load up models
    dotenv.load_dotenv()
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

    # light and heavy LLMs for different tasks
    LM_LIGHT = dspy.LM("gemini/gemini-3-flash-preview", api_key=GEMINI_API_KEY)
    LM_HEAVY = dspy.LM("gemini/gemini-3-pro-preview", api_key=GEMINI_API_KEY)
    EMB = dspy.Embedder("gemini/gemini-embedding-001", api_key=GEMINI_API_KEY)

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

def get_importance(text: str) -> float:
    """
    Evaluate the narrative importance of a given text.

    Args:
        text (str): The text content to evaluate.

    Returns:
        float: A value between 0.0 and 1.0 representing the importance of the text.
    """
    init()
    return importance(text=text, lm=LM_LIGHT).importance

def get_emb(text: typing.Union[str, list[str]]) -> np.ndarray[np.float32]:
    """
    Generate vector embeddings for the provided text.

    Args:
        text (str or list[str]): A single string or a list of strings to embed.

    Returns:
        np.ndarray[np.float32]: The computed embeddings for the text.
    """
    if isinstance(text, str):
        return get_emb([text])[0]

    init()
    
    emb = EMB([t or " " for t in text])
    return np.stack([e if t else np.zeros_like(e) for t, e in zip(text, emb)], axis=0).astype(np.float32)

# Global flag so we only run the download once
_NLTK_INITIALIZED = False

@functools.lru_cache(maxsize=100)
def count_sentences(text: str) -> int:
    """
    Counts how many sentences a text string contains, returning an integer.
    """
    global _NLTK_INITIALIZED
    if not _NLTK_INITIALIZED:
        # Downloads the grammar rules silently
        nltk.download("punkt", quiet=True)
        _NLTK_INITIALIZED = True
        
    return len(nltk.tokenize.sent_tokenize(text))