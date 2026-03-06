import os
import typing

import dotenv
import dspy

GEMINI_API_KEY: str | None = None
LM_LIGHT: dspy.LM | None = None
LM_HEAVY: dspy.LM | None = None
EMB: dspy.Embedder | None = None
importance: dspy.Predict | None = None

def init() -> None:
    """
    Initialize the Worldline library.
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
    EMB = dspy.Embedder("models/text-embedding-004", api_key=GEMINI_API_KEY)

    importance = dspy.Predict(
        dspy.Signature(
            "text: str -> importance: float",
            instructions = """
            Rate the importance or notability of the text and what it describes on a scale from 0 to 100. 
            0 is not important at all and 100 is extremely important. 
            Then divide by 100 to get a value between 0 and 1.
            """
        )
    ).with_config(lm=LM_LIGHT)

def get_importance(text: str) -> float:
    init()
    return importance(text).value

def get_emb(text: str | list[str]) -> list[float] | list[list[float]]:
    init()
    return EMB(text)

class PageCounter:
    """
    Pages are used to measure narrative progression. 
    Each event in Worldline is associated with a page or range of pages.
    """
    def __init__(self, start: int = 0):
        self.page = start

    def step(self) -> int:
        self.page += 1
        return self.page
    
    def __str__(self) -> str:
        return f"Page {self.page}"

    def __repr__(self) -> str:
        return f"PageCounter({self.page})"

class Note:
    def __init__(
        self, 
        name: str, 
        content: str, 
        page: int | list[int] | PageCounter = 0
    ) -> None:
        self.name = name
        self.content = content
        if isinstance(page, int):
            self.pages = [page]
        elif isinstance(page, list):
            self.pages = page
        elif isinstance(page, PageCounter):
            self.pages = [page.page]

    @property
    def emb(self) -> list[float]:
        return get_emb(self.content)

    @property
    def importance(self) -> float:
        return get_importance(self.content)

    def __str__(self) -> str:
        return f"{self.name}: {self.content}"

# contain parameters for Worldline/Library recall and other stuff
class RecallSettings(typing.NamedTuple):
    weight_sim: float = 0.5
    weight_recency: float = 0.25
    weight_importance: float = 0.25
    page_decay: float = 0.99