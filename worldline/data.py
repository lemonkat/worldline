from dataclasses import dataclass, field
import typing
import numpy as np

from worldline.uid import UID

class PageCounter:
    """
    Tracks narrative progression in increments known as 'pages'.

    Pages are used to measure time or flow within the story. Each event 
    in Worldline is typically associated with a specific page or range of
    pages. One page is not guaranteed to represent any specific amount
    of time and are meant primarily for comparison use. 

    Attributes:
        page (int): The current page number.
    """
    def __init__(self, start: int = 0):
        """
        Initialize the PageCounter.

        Args:
            start (int, optional): The initial page number. Defaults to 0.
        """
        self.page = start

    def step(self) -> int:
        """
        Advance the page counter by one.

        Returns:
            int: The new current page number.
        """
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


@dataclass
class Note:
    """Base class for all vector-embeddable knowledge in the Worldline system."""
    id: UID
    content: str
    page: int
    
    # Defaults to None so they can be lazily fetched using __post_init__ if desired,
    # or passed in manually to skip the API call.
    importance: typing.Optional[float] = field(default=None)
    emb: typing.Optional[np.ndarray] = field(default=None)

    def get_embed_text(self) -> str:
        """Subclasses can override this to embed different text!"""
        return self.content
    
    def __post_init__(self):
        from worldline.llm import get_emb, get_importance
        # We call these methods dynamically instead of hardcoding self.content
        text = self.get_embed_text()
        if self.importance is None:
            self.importance = get_importance(text)
        if self.emb is None:
            self.emb = get_emb(text)
