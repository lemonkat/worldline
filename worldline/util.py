import os
import typing
import dataclasses
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

@functools.cache
def get_importance(text: str) -> float:
    init()
    return importance(text=text, lm=LM_LIGHT).importance

@functools.cache
def get_emb(text: str | list[str]) -> np.ndarray[np.float32]:
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
    
    def __index__(self) -> int:
        return self.page

    def __int__(self) -> int:
        return self.page

@dataclasses.dataclass(frozen=True)
class Note:
    """
    Notes are the fundamental unit of information in Worldline.

    A note is a piece of information, sometimes associated with a specific page.
    page -1 is for "timeless" information that is not associated with a specific page or point in time.
    """
    name: str
    content: str
    page: int | PageCounter = -1

    def __post_init__(self):
        object.__setattr__(self, "page", int(self.page))
        
    @property
    def emb(self) -> np.ndarray[np.float32]:
        return get_emb(str(self))

    @property
    def importance(self) -> float:
        return get_importance(self.content)

    def __str__(self) -> str:
        return f"{self.name}: {self.content} (page {self.page})" if self.page != -1 else f"{self.name}: {self.content}"

    def copy_with(
        self, 
        name: str | None = None, 
        content: str | None = None, 
        page: int | PageCounter | None = None
    ) -> typing.Self:
        return dataclasses.replace(
            self, 
            name=self.name if name is None else name, 
            content=self.content if content is None else content, 
            page=self.page if page is None else page
        )

    