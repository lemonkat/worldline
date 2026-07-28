"""
[Phase Two: Libraries]

Databases storing unorganized information for Project Worldline, suited for general world info or long-term memories.

This module implements the Heap portion of the Fate Engine's memory architecture.
Libraries are collections of Record objects with search-via-ID, 
search-via-vector-embedding, and create-read-update-delete capabilities.
The context window loading mechanism is fully stateless between turns, designed specifically 
for an Auto-RAG (Retrieval-Augmented Generation) loop where relevant memories are pulled
at the start of every cognitive cycle.
"""

import typing
import heapq

from pydantic import Field, ConfigDict, PrivateAttr
import numpy as np
import dspy

from worldline.uid import UID
from worldline.llm import get_emb, count_sentences
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
        lines = [f"[Record] {self.title}:", self.content]
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

        Should be called by before any ReAct agents are given tool access.
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
        lines = [f"[Library] {self.title}"]
        lines.extend(record.get_content() for record in records)
        return "\n".join(lines)

    def _get_content(self) -> str:
        """Generates the context window prompt for the Library.

        Sorts all currently loaded Records by their relevance scores (highest first) 
        and formats them into a single string for injection into the DSPy prompt.

        Returns:
            str: The formatted working memory of the Library.
        """
        return self.format_records(self.records[uid] for uid in sorted(list(self.loaded.keys()), key=self.loaded.get, reverse=True))


    def _tool_recall(self, uid: UID) -> str:
        """DSPy tool wrapper for manually recalling a specific Record by UID.
        
        Args:
            uid (UID): The unique identifier of the Record to recall.
            
        Returns:
            str: A success message with the formatted Record, or an error string.
        """
        if uid not in self.records:
            return f"Error: No record with UID {uid} could be found. It may have been deleted."
        return f"Success. Found record: {self.format_records([self.recall(uid)])}"

    @property
    def tool_recall(self) -> dspy.Tool:
        return dspy.Tool(
            self._tool_recall,
            f"{self.title} - Recall",
            "Finds a Record given its exact UID.",
            arg_desc={
                "uid": "The exact UID of the record. Case-sensitive.",
            }
        )

    def _tool_search(self, query: str) -> str:
        """DSPy tool wrapper for performing an Auto-RAG semantic search.
        
        Args:
            query (str): The search query to embed.
            
        Returns:
            str: A success message with the formatted highest-scoring novel Records.
        """
        return f"Success. Found records: {self.format_records(self.search(query, True))}"

    @property
    def tool_search(self) -> dspy.Tool:
        return dspy.Tool(
            self._tool_search,
            f"{self.title} - Search",
            f"Find {self.ctx.config.library_search_k} Records via cosine similarity of embedding vectors. Will not return any Records that are already in the context.",
            arg_desc={
                "query": "The query to use for the search.",
            }
        )

    def _tool_create(self, title: str, content: str, source: str, related: str) -> str:
        """DSPy tool wrapper for creating a new Record in the Library.
        
        Args:
            title (str): The title of the new Record.
            content (str): The narrative text.
            source (str): A description of the origin of this memory ('N/A' for None).
            related (str): A comma-separated string of related UIDs ('N/A' for None).
            
        Returns:
            str: A success message with the formatted Record, or an error string.
        """
        n_sentences = count_sentences(content)
        if n_sentences > self.ctx.config.library_max_record_size:
            return f"Error: Content is too long ({n_sentences} sentences). Max length is {self.ctx.config.library_max_record_size} sentences. No changes have been made."
        related = related.upper().strip()
        related_UIDs = [] if related == "N/A" else [uid.strip() for uid in related.split(",") if uid.strip()]
        for uid in related_UIDs:
            if uid not in self.records:
                return f"Error: Referenced UID {uid} but no Record with that UID could be found. It may have been deleted. No changes have been made."
        if len(related_UIDs) > self.ctx.config.library_max_n_refs:
            return f"Error: Too many Records referenced ({len(related_UIDs)} Records). Max is {self.ctx.config.library_max_n_refs} Records."
        record = self.create(title, content, None if source.upper().strip() == "N/A" else source, related_UIDs)
        return f"Success. Created Record: {self.format_records([record])}"

    @property
    def tool_create(self) -> dspy.Tool:
        return dspy.Tool(
            self._tool_create,
            f"{self.title} - Create",
            "Create a new Record.",
            arg_desc={
                "title": "A title for this Record.",
                "content": f"The contents of this Record. Recommended {self.ctx.config.library_avg_record_size} sentences, maximum {self.ctx.config.library_max_record_size} sentences.",
                "source": "A very short description of how this information came to be. Write 'N/A' to leave empty.",
                "related": f"UIDs of potentially relevant existing Records. Case-sensitive. Separate by commas, like 'XY12, A1B9, 74J8'. Write 'N/A' to leave empty. Recommended {self.ctx.config.library_avg_n_refs} references, max {self.ctx.config.library_max_n_refs} references.",
            }
        )
    
    def _tool_update(self, uid: UID, title: str, content: str, source: str, related: str, append: bool) -> str:
        """DSPy tool wrapper for modifying an existing Record in working memory.
        
        Args:
            uid (UID): The UID of the Record to modify.
            title (str): The new title ('N/A' to skip).
            content (str): The new text content ('N/A' to skip).
            source (str): The new origin description ('N/A' to skip).
            related (str): A comma-separated string of new related UIDs ('N/A' to skip).
            append (bool): If True, appends the new content instead of overwriting.
            
        Returns:
            str: A success message with the formatted Record, or an error string.
        """
        if uid not in self.records:
            return f"Error: No Record with UID {uid} could be found. It may have been deleted. No changes have been made."
        if uid not in self.loaded:
            return f"Error: The Record with UID {uid} is not in the context. Recall it first to know what you are updating. No changes have been made."


        if append and content.upper() != "N/A":
            content = self.records[uid].content.strip() + "\n" + content

        n_sentences = count_sentences(content)
        if n_sentences > self.ctx.config.library_max_record_size:
            return f"Error: Content is too long ({n_sentences} sentences). Max length is {self.ctx.config.library_max_record_size} sentences. Consider setting append=False and summarizing existing content. No changes have been made."

        related = related.upper().strip()
        if related == "N/A":
            related_UIDs = None
        else:
            related_UIDs = [uid.strip() for uid in related.split(",") if uid.strip()]
            for uid in related_UIDs:
                if uid not in self.records:
                    return f"Error: Referenced UID {uid} but no Record with that UID could be found. It may have been deleted. No changes have been made."
            if len(related_UIDs) > self.ctx.config.library_max_n_refs:
                return f"Error: Too many Records referenced ({len(related_UIDs)} Records). Max is {self.ctx.config.library_max_n_refs} Records."
        record = self.update(
            uid, 
            None if title.upper() == "N/A" else title,
            None if content.upper() == "N/A" else content,
            None if source.upper() == "N/A" else source,
            related_UIDs,
            False,
        )
        return f"Success. Updated Record: {self.format_records([record])}"

    @property
    def tool_update(self) -> dspy.Tool:
        return dspy.Tool(
            self._tool_update,
            f"{self.title} - Update",
            "Update an existing Record.",
            arg_desc={
                "uid": "The UID of the Record to edit. Case-sensitive. Must be a Record currently in the context.",
                "title": "The new title for this Record. Write 'N/A' to not edit. Avoid editing this unless necessary.",
                "content": f"The contents of this Record. Recommended {self.ctx.config.library_avg_record_size} sentences, maximum {self.ctx.config.library_max_record_size} sentences. Write 'N/A' to not edit.",
                "source": "A very short description of how this information came to be. Write 'N/A' to not edit. Avoid editing this unless necessary. Will overwrite existing sources.",
                "related": f"UIDs of potentially relevant existing Records. Case-sensitive. Separate by commas, like 'XY12, A1B9, 74J8'. Write 'N/A' to not edit. Avoid editing this unless necessary. Will overwrite existing references. Recommended {self.ctx.config.library_avg_n_refs} references, max {self.ctx.config.library_max_n_refs} references.",
                "append": "Whether or not to append to the existing content or overwrite it. Does nothing if content = 'N/A'.",
            }
        )

    def _tool_delete(self, uid: UID) -> str:
        """DSPy tool wrapper for deleting an existing Record from the Library.
        
        Args:
            uid (UID): The UID of the Record to delete.
            
        Returns:
            str: A success message, or an error string.
        """
        if uid not in self.records:
            return f"Error: No Record with UID {uid} could be found. It may have been deleted. No changes have been made."
        if uid not in self.loaded:
            return f"Error: The Record with UID {uid} is not in the context. Recall it first to know what you are deleting. No changes have been made."

        self.delete(uid)
        return "Success. Record deleted."

    @property
    def tool_delete(self) -> dspy.Tool:
        return dspy.Tool(
            self._tool_delete,
            f"{self.title} - Delete",
            "Delete an existing Record.",
            arg_desc={
                "uid": "The UID of the Record to delete. Case-sensitive. Must be a Record currently in the context.",
            }
        )

    @property
    def tools(self) -> list[dspy.Tool]:
        """Returns a list of DSPy tools exposed by this object."""
        return [self.tool_recall, self.tool_search, self.tool_create, self.tool_update, self.tool_delete]