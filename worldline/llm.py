import os
import functools

import numpy as np
    
import dotenv
import dspy

GEMINI_API_KEY: str | None = None
LM_LIGHT: dspy.LM | None = None
LM_HEAVY: dspy.LM | None = None
EMB: dspy.Embedder | None = None
importance: dspy.Predict | None = None

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

@functools.lru_cache(maxsize=100)
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

@functools.lru_cache(maxsize=100)
def get_emb(text: str | list[str]) -> np.ndarray[np.float32]:
    """
    Generate vector embeddings for the provided text.

    Args:
        text (str | list[str]): A single string or a list of strings to embed.

    Returns:
        np.ndarray[np.float32]: The computed embeddings for the text.
    """
    init()
    return EMB(text)