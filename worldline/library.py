"""
[Phase Two: Libraries]

Databases storing unorganized information for Project Worldline, suited for general world info or long-term memories.

This module implements the Heap portion of the Fate Engine's memory architecture.
Libraries are collections of Record objects with search-via-ID, 
search-via-vector-embedding, and create-read-update-delete capabilities.
The context window loading mechanism is fully stateless between turns, designed specifically 
for an Auto-RAG (Retrieval-Augmented Generation) loop where the Orchestrator pulls relevant 
memories at the start of every cognitive cycle.
"""

import typing
import heapq

from pydantic import Field, ConfigDict, PrivateAttr
import numpy as np

from worldline.uid import UID
from worldline.llm import get_emb
from worldline.data import Note

class MissingRecordError(Exception):
    """Raised when the system attempts to access a nonexistent Record."""
    pass

class UnloadedRecordError(Exception):
    """Raised when the system attempts to access a Record not in working memory."""
    pass

class Record(Note):
    """A discrete unit of semantic memory for long-term storage in a Library.

    Records represent specific facts, beliefs, or memories that a Persona holds.
    They include provenance tracking (source) and an associative graph of 
    pointers to other records (related).

    Attributes:
        title (str): A short, descriptive title for the Record.
        content (str): The detailed text of the Record.
        source (str | None): A text description of the epistemological origin of the Record.
        related (list[UID]): A list of UIDs pointing to other associated Records.
    """
    title: str = "UNTITLED"
    content: str = ""
    source: typing.Optional[str] = None
    related: list[UID] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
        
    # Private cache fields. Use the properties `importance` and `emb` instead.
    # These are only passed in manually when loading from a save file on disk.
    _importance: typing.Optional[float] = PrivateAttr(default=None)
    _emb: typing.Optional[np.ndarray] = PrivateAttr(default=None)

    @property
    def importance(self) -> float:
        """Lazily evaluates and returns the importance score of the Note.

        Returns:
            float: The importance score.
        """
        if self._importance is None:
            from worldline.llm import get_importance
            self._importance = get_importance(self.get_content(False))
        return self._importance

    @property
    def emb(self) -> np.ndarray:
        """Lazily evaluates and returns the vector embedding of the Note.

        Returns:
            np.ndarray: The embedding vector.
        """
        if self._emb is None:
            from worldline.llm import get_emb
            self._emb = get_emb(self.get_content(False))
        return self._emb


    def _get_content(self) -> str:
        """Formats the Record for the LLM context window.

        Dynamically resolves the titles of any related UIDs from the global 
        registry to provide semantic hints to the LLM.

        Returns:
            str: The formatted narrative string representing this memory.
        """
        lines = [f"{self.title}:", self.content]
        if self.source: 
            lines.append(f"Source: {self.source}")
        if self.related:
            related_titles = [f"[UID {uid}] {self.ctx.registry[uid].title}" for uid in self.related]
            lines.append("Related: " + ", ".join(related_titles))
        return "\n".join(lines)
    
    def _pack(self) -> dict:
        """Serializes the Record's specific fields into a dictionary.

        Returns:
            dict: The state dictionary.
        """
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "related_UIDs": self.related
        }

    def _unpack(self, state: dict) -> None:
        """Restores the Record's fields from a state dictionary.

        Args:
            state (dict): The dictionary containing the saved state.
        """
        self.title = state["title"]
        self.content = state["content"]
        self.source = state["source"]
        self.related = state["related_UIDs"]

        self._importance = None
        self._emb = None

class Library(Note):
    """A collection of long-term semantic memories (Records).

    Libraries track their own working memory via the `loaded` dictionary, which maps 
    currently active UIDs to their relevance scores. This architecture enforces that 
    the LLM is only authorized to read or mutate Records that are explicitly loaded 
    into its context during the current turn.

    Attributes:
        title (str): The name of the Library (e.g., "World Knowledge", "Bob's Beliefs").
        records (dict[UID, Record]): The master database of all contained records.
        loaded (dict[UID, float]): Working memory mapping active UIDs to relevance scores.
    """
    title: str = "MAIN"
    records: dict[UID, Record] = Field(default_factory=dict)
    loaded: dict[UID, float] = Field(default_factory=dict)

    def _pack(self) -> dict:
        """Serializes the Library's specific fields into a dictionary.

        Returns:
            dict: The state dictionary.
        """
        return {
            "record_UIDs": list(self.records.keys()),
            "loaded_UIDs": self.loaded,
        }

    def _unpack(self, state: dict) -> None:
        """Restores the Library's fields from a state dictionary.

        Args:
            state (dict): The dictionary containing the saved state.
        """
        self.records = {uid: self.ctx.registry[uid] for uid in state["record_UIDs"]}
        self.loaded = state["loaded_UIDs"]

    def refresh(self) -> None:
        """Clears the working memory.

        Should be called by the Orchestrator at the start of every cognitive turn 
        before Auto-RAG, ensuring the LLM context remains perfectly temporally relevant.
        """
        self.loaded = {}

    def recall(self, uid: UID) -> Record:
        """Manually loads a specific Record into working memory.

        Used when the LLM explicitly requests a Record by its UID. Assigns a massive 
        artificial score (1e6) to guarantee it sorts above any semantic Auto-RAG results.

        Args:
            uid (UID): The unique identifier of the Record to load.

        Returns:
            Record: The requested Record.

        Raises:
            MissingRecordError: If the UID does not exist in the Library.
        """
        if uid not in self.records:
            raise MissingRecordError(f"No record found with UID {uid}.")
        if uid not in self.loaded:
            self.loaded[uid] = 1e6 # in case someone sets the weights high, this will probably be higher still
            self.edited = True
        return self.records[uid]

    def search(self, query: str, mark_loaded: bool = False) -> list[Record]:
        """Performs a semantic and importance-weighted Auto-RAG search.

        Calculates a blended score based on cosine similarity to the query and the Record's 
        intrinsic importance, filtering out any Records already present in working memory.

        Args:
            query (str): The string to embed and search against.
            mark_loaded (bool, optional): If True, adds the found Records to working memory 
                with their exact blended scores. Defaults to False.

        Returns:
            list[Record]: The highest scoring novel Records, up to `library_search_k`.
        """
        query_emb = get_emb(query)
        scores = []
        for uid, record in self.records.items():
            if uid in self.loaded:
                continue
            score_sim = float(np.dot(query_emb, record.emb))
            score_imp = record.importance
            scores.append((score_sim * self.ctx.config.library_search_weight_sim + score_imp * self.ctx.config.library_search_weight_imp, uid))
        found = {uid: score for score, uid in heapq.nlargest(self.ctx.config.library_search_k, scores)}
        if mark_loaded:
            self.loaded.update(found)
            self.edited = True
        return [self.records[uid] for uid in found]

    def create(
        self,
        title: str,
        content: str,
        source: typing.Optional[str] = None,
        related: typing.Optional[list[UID]] = None, 
    ) -> Record:
        """Creates a new Record and stores it in the Library.

        Args:
            title (str): The short title of the Record.
            content (str): The detailed text.
            source (str | None, optional): Provenance description. Defaults to None.
            related (list[UID] | None, optional): Pointers to associated Records. Defaults to None.

        Returns:
            Record: The newly instantiated Record.
        """
        record = Record(ctx=self.ctx, title=title, content=content, source=source, related=related or [])
        self.records[record.uid] = record
        self.edited = True
        return record

    def update(
        self, 
        uid: UID, 
        title: typing.Optional[str] = None, 
        content: typing.Optional[str] = None,
        source: typing.Optional[str] = None,
        related: typing.Optional[list[UID]] = None, 
        append: bool = True,
    ) -> Record:
        """Mutates an existing Record, provided it is currently loaded in working memory.

        Args:
            uid (UID): The UID of the Record to modify.
            title (str | None, optional): New title. Defaults to None.
            content (str | None, optional): New content text. Defaults to None.
            source (str | None, optional): New source description. Defaults to None.
            related (list[UID] | None, optional): New associative pointers. Defaults to None.
            append (bool, optional): If True, appends the new content to the old content. 
                If False, overwrites the content. Defaults to True.

        Returns:
            Record: The modified Record.

        Raises:
            UnloadedRecordError: If the Record is not currently loaded in working memory.
        """
        self._verify_loaded(uid)
        record = self.records[uid]
        if title is not None:
            record.title = title
        if content is not None:
            record.content = record.content + content if append else content
        if source is not None:
            record.source = source
        if related is not None:
            record.related = related

        self.edited = True
        record.edited = True
        return record

    def delete(self, uid: UID) -> None:
        """Permanently removes a Record from the Library.

        Args:
            uid (UID): The UID of the Record to delete.

        Raises:
            UnloadedRecordError: If the Record is not currently loaded in working memory.
        """
        self._verify_loaded(uid)
        del self.records[uid]
        self.edited = True

    def _verify_loaded(self, uid: UID) -> None:
        """Internal helper to enforce read/write authorization.

        Raises:
            MissingRecordError: If the Record doesn't exist.
            UnloadedRecordError: If the Record is not loaded.
        """
        if uid not in self.records:
            raise MissingRecordError(f"No record found with UID {uid}.")
        elif uid not in self.loaded:
            raise UnloadedRecordError(f"The record with UID {uid} is not loaded in working memory.")

    def format_records(self, records: list[Record]) -> str:
        """Helper to format a list of Records for the LLM context window.

        Args:
            records (list[Record]): The Records to format.

        Returns:
            str: The concatenated narrative string.
        """
        lines = [f"Library: {self.title}"] + [record.get_content() for record in records]
        return "\n".join(lines)

    def _get_content(self) -> str:
        """Generates the context window prompt for the Library.

        Sorts all currently loaded Records by their relevance scores (highest first) 
        and formats them into a single string for injection into the DSPy prompt.

        Returns:
            str: The formatted working memory of the Library.
        """
        return self.format_records(self.records[uid] for uid in sorted(list(self.loaded.keys()), key=self.loaded.get, reverse=True))
