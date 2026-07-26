"""
[Phase Zero: supporting infrastructure]

Core data structures and domain models for the Worldline system.

This module defines the foundational classes for representing narrative time
and embeddable knowledge entries, utilizing lazy evaluation 
for LLM-derived attributes to avoid circular dependencies.
"""

from dataclasses import dataclass, field
import typing

import numpy as np

from worldline.uid import UID, UIDGenerator

# pages, meant for comparison and undo/redo but not necessarily arithmetic
Page = int

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
    def __init__(self, start: Page = 0):
        """
        Initialize the PageCounter.

        Args:
            start (int, optional): The initial page number. Defaults to 0.
        """
        self.page = start

    def step(self) -> Page:
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
class Config:
    """
    Bundled configuration settings for Worldline.
    """
    worldline_recall_k: int = 100
    worldline_max_depth: int = 10
    # text sizes measured in sentences
    worldline_max_entry_size: int = 10

# note: uid generator and page counter are mutable
@dataclass(frozen=True)
class Context:
    """
    Bundled state tracking objects for Worldline.
    """
    uid_generator: UIDGenerator
    page_counter: PageCounter
    config: Config


@dataclass(eq=False)
class Note:
    """
    Base class for all vector-embeddable knowledge in the Worldline system.

    This class serves as the foundation for both the narrative Stack (WorldlineFrames) 
    and the narrative Heap (LibraryMemories). It natively implements the Memento pattern 
    for time-travel via Bi-Temporal history tracking, allowing objects to perfectly 
    revert to past states without destroying future timelines.

    It also guarantees that every entry has a unique ID, a timestamp, and 
    automatically synchronizes its LLM embeddings whenever its state changes.
    """

    class NoteState(typing.NamedTuple):
        """A snapshot of the Note's state at a specific point in narrative time."""
        page: Page
        state: str

    context: Context
    id: typing.Optional[UID] = None
    
    # Private cache fields. Use the properties `importance` and `emb` instead.
    # These are only passed in manually when loading from a save file on disk.
    _importance: typing.Optional[float] = field(default=None)
    _emb: typing.Optional[np.ndarray] = field(default=None)

    # Time-travel log mapping page numbers to packed string states
    hist: list[NoteState] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """
        Initializes identity for fresh objects.
        If an object is loaded from disk, the existing ID is safely preserved.
        """
        if self.id is None:
            self.id = self.context.uid_generator.next()

    @property
    def importance(self) -> float:
        """Lazily evaluates the importance score of the Note."""
        if self._importance is None:
            from worldline.llm import get_importance
            self._importance = get_importance(self.get_content(False))
        return self._importance

    @property
    def emb(self) -> np.ndarray:
        """Lazily evaluates the vector embedding of the Note."""
        if self._emb is None:
            from worldline.llm import get_emb
            self._emb = get_emb(self.get_content(False))
        return self._emb

    def get_content(self, include_id: bool = True) -> str:
        """
        Public API: Returns the narrative text of this Note for LLM contexts.
        
        Args:
            include_id (bool): If True, prepends the Note's UID to the string.
        """
        if include_id:
            return f"[ID {self.id}] {self._get_content()}"
        return self._get_content()

    def _get_content(self) -> str:
        """
        Private Implementation: Subclasses MUST override this to return their narrative text.
        """
        return ""

    def unpack(self, state: str) -> None:
        """
        Public API: Restores the Note to a previous state and synchronizes embeddings.
        
        Args:
            state (str): The packed string representation to restore.
        """
        if state != self.pack():
            self._unpack(state)
            
            # Clear caches so the properties lazily fetch the new state when accessed!
            self._importance = None
            self._emb = None

    def _unpack(self, state: str) -> None:
        """
        Private Implementation: Subclasses MUST override this to apply a packed state string
        to their internal variables.
        """
        return

    def pack(self) -> str:
        """
        Public API: Packages the Note's current narrative state into a string snapshot.
        """
        return self._pack()

    def _pack(self) -> str:
        """
        Private Implementation: Subclasses MUST override this to return a string representation
        of their current narrative variables (ignoring context and embeddings).
        """
        return ""

    def sync(self) -> None:
        """
        Time-Travel: Reverts the Note's internal variables to match the global clock.
        This does NOT erase future history, allowing for perfect Undo/Redo tracking.
        """
        page = self.context.page_counter.page
        valid_states = [s.state for s in self.hist if s.page <= page]
        
        if valid_states:
            self.unpack(valid_states[-1])

    def save(self) -> None:
        """
        Time-Travel: Logs the current state to history. If the Note has diverged from
        the established timeline, this safely erases the canceled future.
        """
        page = self.context.page_counter.page
        state = self.pack()
        
        while self.hist and self.hist[-1].page >= page:
            self.hist.pop()
            
        if not self.hist or self.hist[-1].state != state:
            self.hist.append(self.NoteState(page, state))

    def __eq__(self, other):
        """
        Equality override to prevent Numpy 'ambiguous truth value' crashes.
        Two notes are identical if they share the same Unique ID.
        """
        if not isinstance(other, Note):
            return False
        return self.id == other.id
            
