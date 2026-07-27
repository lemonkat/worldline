"""
[Phase Two: Libraries]

Databases storing unorganized information for Project Worldline, suited for general world info or long-term memories.

This module implements the Heap portion of the Fate Engine's memory architecture.
Libraries are collections of Record objects with search-via-ID, 
search-via-vector-embedding, and create-read-update-delete capabilities.
"""

import typing

from pydantic import Field

from worldline.uid import UID
from worldline.data import Note

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

