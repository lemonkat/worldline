"""
[Phase Zero: supporting infrastructure]

Core data structures and domain models for the Project Worldline system.

This module defines the foundational classes for representing narrative time
and embeddable knowledge entries, utilizing lazy evaluation 
for LLM-derived attributes to avoid circular dependencies.
"""

from dataclasses import dataclass, field
import typing
import threading

from pydantic import BaseModel, model_validator, ConfigDict
import dspy

from worldline.uid import UID, UIDGenerator

# pages, meant for comparison and undo/redo but not necessarily arithmetic
Page = int

class PageCounter:
    """
    Tracks narrative progression in increments known as 'pages'.

    Pages are used to measure time or flow within the story. Each event 
    in Worldline is typically associated with a specific page or range of
    pages. Pages are not guaranteed to represent any specific amount
    of time and are meant primarily for comparison use. 

    Attributes:
        page (Page): The current page number.
    """
    def __init__(self, start: Page = 0):
        """
        Initialize the PageCounter.

        Args:
            start (Page, optional): The initial page number. Defaults to 0.
        """
        self.page = start

    def step(self) -> Page:
        """
        Advance the page counter by one.

        Returns:
            Page: The new current page number.
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

class Config(BaseModel):
    """Bundled configuration settings for Worldline.

    Attributes:
        worldline_recall_k (int): How many recent beats/sub-arcs per arc to recall for non-root Worldlines.
        worldline_max_depth (int): Maximum nesting depth for Worldlines.
        worldline_avg_entry_size (int): Average text size measured in sentences for Worldline entry contents.
        worldline_max_entry_size (int): Maximum text size measured in sentences for Worldline entry contents.

        library_search_k (int): The number of novel records returned by an Auto-RAG search.
        library_search_weight_sim (float): The multiplier for semantic cosine similarity when searching.
        library_search_weight_imp (float): The multiplier for intrinsic importance when searching.
        library_avg_record_size (int): Average text size measured in sentences for a Record.
        library_max_record_size (int): Maximum text size measured in sentences for a Record.
        library_avg_n_refs (int): Average number of related pointers a Record should maintain.
        library_max_n_refs (int): Maximum number of related pointers a Record should maintain.
    """
    worldline_recall_k: int = 10
    worldline_max_depth: int = 10
    worldline_avg_entry_size: int = 3
    worldline_max_entry_size: int = 10

    library_search_k: int = 10
    library_search_weight_sim: float = 0.5
    library_search_weight_imp: float = 0.5
    library_avg_record_size: int = 8
    library_max_record_size: int = 20
    library_avg_n_refs: int = 2
    library_max_n_refs: int = 5

@dataclass
class Context:
    """Bundled state tracking objects for Worldline.

    Maintainins the global Note registry and the delta-log history.

    Attributes:
        uid_generator (UIDGenerator): The generator for unique IDs.
        page_counter (PageCounter): The tracker for narrative time.
        config (Config): The global configuration settings.
        registry (dict[UID, Note]): The global dictionary of all instantiated entities.
        history (list[Update]): The flat-list delta log of all state changes.
    """
    class Update(typing.NamedTuple):
        """A lightweight, immutable record of a Note's state at a specific page.

        Attributes:
            page (Page): The page number when this state was recorded.
            uid (UID): The unique identifier of the edited note.
            state (dict): The packed dictionary snapshot of the note.
        """
        page: Page
        uid: UID
        state: dict

    uid_generator: UIDGenerator
    page_counter: PageCounter
    config: Config
    registry: dict[UID, "Note"] = field(default_factory=dict)
    history: list[Update] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def record(self) -> None:
        """Commits all current edits to the history log.

        Scans the registry for edited entities, packs their state, and appends 
        them to the history log. If the current page is strictly less than the 
        latest history entry, the future timeline is erased.
        """
        with self.lock:
            cur_page = self.page_counter.page
            while self.history and self.history[-1].page > cur_page:
                self.history.pop()

            for entity in self.registry.values():
                if entity.edited:
                    entity.edited = False
                    self.history.append(self.Update(cur_page, entity.uid, entity.pack()))
        

    def rewind(self) -> None:
        """Reverts the game state to the current page.

        Reverts all entities in the registry to their most recent state prior 
        to or equal to the current page. Entities created after the current 
        page are safely ignored.
        """
        with self.lock:
            cur_page = self.page_counter.page
            updated = set()
            for page, uid, state in reversed(self.history):
                if page > cur_page or uid in updated:
                    continue
                updated.add(uid)
                self.registry[uid].unpack(state)
                if len(updated) == len(self.registry):
                    break

class Note(BaseModel):
    """Base class for all saveable objects in the Worldline system.

    Provides core infrastructure for generating unique IDs, registering the object 
    with the global Context, and handling pack/unpack serialization.

    Attributes:
        ctx (Context): The global engine context.
        uid (UID, optional): The universally unique identifier for this note. Defaults to creating a new UID from the context's UIDGenerator.
        edited (bool): Flag indicating if the note was mutated recently.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ctx: Context
    uid: typing.Optional[UID] = None
    edited: bool = True
    lock: threading.RLock = field(default_factory=threading.RLock)

    sys_name: typing.ClassVar[str] = "Note"
    sys_desc: typing.ClassVar[str] = "Base generic note."

    @model_validator(mode="after")
    def _setup(self) -> "Note":
        """Validates the Note post-initialization.

        Assigns a unique ID if one was not provided, and registers the note 
        in the global context registry.

        Returns:
            Note: The validated and registered note.
        """
        if self.uid is None:
            self.uid = self.ctx.uid_generator.next()
        self.ctx.registry[self.uid] = self
        return self

    def unpack(self, state: dict) -> None:
        """Restores the Note to a previous state.

        Public API for applying a time-travel or save-file snapshot.

        Args:
            state (dict): The packed dictionary representation to restore.
        """
        if state != self.pack():
            self._unpack(state)

    def _unpack(self, state: dict) -> None:
        """Applies a packed state dictionary to internal variables.

        Private implementation. Subclasses MUST override this.

        Args:
            state (dict): The packed dictionary representation.
        """
        return

    def pack(self) -> dict:
        """Packages the Note's current narrative state into a snapshot.

        Public API for generating a state dictionary for the Delta Log.

        Returns:
            dict: The serialized state.
        """
        return self._pack()

    def _pack(self) -> dict:
        """Returns a dictionary representation of current narrative variables.

        Private implementation. Subclasses MUST override this. Context and 
        embeddings should be ignored.

        Returns:
            dict: The serialized state.
        """
        return {}

    def __eq__(self, other) -> bool:
        """Checks equality against another object.

        Overrides standard equality to prevent Numpy 'ambiguous truth value' crashes.
        Two Notes are identical if they share the same Unique ID.

        Args:
            other (Any): The object to compare against.

        Returns:
            bool: True if the UIDs match, False otherwise.
        """
        if not isinstance(other, Note):
            return False
        return self.uid == other.uid

    def get_content(self, include_uid: bool = True, **kwargs: dict) -> str:
        """Returns the narrative text of this Note for LLM contexts.
        
        Args:
            include_uid (bool, optional): If True, prepends the Note's UID. Defaults to True.
            **kwargs: Arbitrary keyword arguments passed to the subclass's `_get_content` implementation.

        Returns:
            str: The formatted narrative string.
        """
        if include_uid:
            return f"[UID {self.uid}] {self._get_content(**kwargs)}"
        return self._get_content(**kwargs)

    def _get_content(self) -> str:
        """Returns the raw narrative text of this Note.

        Private implementation. Subclasses MUST override this.

        Returns:
            str: The narrative text.
        """
        return "[Note]"

    @property
    def tools(self) -> list[dspy.Tool]:
        """Returns a list of DSPy tools exposed by this object."""
        return []

    def initialize(self, context: str = "", **kwargs: dict) -> str:
        """Prepares this Note to be used by an Agent using the given context, then returns `self.get_content(**kwargs)`.

        Should be called by before any ReAct agents are given tool access.

        Args:
            context (str, optional): Context to be used for initialization. Defaults to the empty string.
            **kwargs: Arbitrary keyword arguments passed to `get_content`.

        Returns:
            str: The formatted narrative string from `self.get_content()`.
        """
        self._initialize(context)
        return self.get_content(**kwargs)

    def _initialize(self, context: str) -> None:
        """Prepares this Note to be used by an Agent using the given context.

        Should be called by before any ReAct agents are given tool access.
        Private implementation. 

        Args:
            context (str): Context to be used for initialization.
        """
        return

            
