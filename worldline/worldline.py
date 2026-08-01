"""
[Phase One: Worldlines]

Call-stack-inspired data structures for recent story events and short-term thought processes in Project Worldline.

This module implements the Stack portion of the Fate Engine's memory architecture.
Worldline frames are hierarchical, recursive Note objects that represent ongoing narrative Arcs 
and atomic Beats, allowing the LLM to easily manage highly contextual, nested narrative scopes.
"""
import typing
import warnings

from pydantic import Field
import dspy

from worldline.uid import UID
from worldline.llm import count_sentences
from worldline.data import Note

class Worldline(Note):
    """A recursive narrative container representing the active story stack.

    A Worldline acts as either a Beat (a leaf node representing a single action with raw content) 
    or an Arc (a parent frame grouping multiple children, where 'content' acts as the summary of those children).
    
    The tree implements an automatic routing architecture: operations called on the Root 
    are mathematically guaranteed to cascade down to the deepest open Arc.

    Attributes:
        name (str): The name of the Arc or Beat.
        content (str, optional): The raw text or summary content. None if the Arc is open.
        children (list[UID]): The ordered list of UIDs of nested Worldlines.
    """
    name: str = "Unnamed Worldline"
    content: typing.Optional[str] = None
    children: list[UID] = Field(default_factory=list)

    sys_name: typing.ClassVar[str] = "Worldline"
    sys_desc: typing.ClassVar[str] = "A hierarchical stack representing recent events or thoughts. You can append atomic Beats, Dive into new nested Arcs to handle sub-tasks, and Surface to summarize completed Arcs."

    def beat(self, name: str, content: str) -> None:
        """Adds a single, atomic Beat node to the deepest open Arc.
        
        Args:
            name (str): A short description of the action.
            content (str): The raw narrative text of the event.
            
        Raises:
            RuntimeError: If the Worldline (or the deepest Arc) is already closed.
        """
        with self.lock:
            if not self.open:
                raise RuntimeError("RuntimeError: Cannot append to a " + ("closed arc" if self.children else "beat") + ".")
            if self.children and not self[-1].content:
                self[-1].beat(name, content)
            else:
                self.children.append(Worldline(ctx=self.ctx, name=name, content=content).uid)
                self.edited = True

    def dive(self, name: str) -> None:
        """Opens a new nested Arc within the deepest open Arc.
        
        Args:
            name (str): The name of the new Arc.
            
        Raises:
            RuntimeError: If the Worldline (or the deepest Arc) is already closed.
        
        Warns:
            RuntimeWarning: If opening the new Arc would exceed worldline_max_depth.
        """
        with self.lock:
            if not self.open:
                raise RuntimeError("RuntimeError: Cannot append to a " + ("closed arc" if self.children else "beat") + ".")
            
            if self.depth >= self.ctx.config.worldline_max_depth:
                warnings.warn(f"Diving past maximum depth of {self.ctx.config.worldline_max_depth}.", RuntimeWarning)
            
            if self.children and not self[-1].content:
                self[-1].dive(name)
            else:
                self.children.append(Worldline(ctx=self.ctx, name=name).uid)
                self.edited = True

    def surface(self, content: str) -> None:
        """Closes the deepest open Arc by populating its summary content.
        
        Args:
            content (str): The LLM-generated summary of all events that occurred inside the Arc.
            
        Raises:
            RuntimeError: If the Worldline is already closed, meaning there are no open Arcs left.
        """
        with self.lock:
            if not self.open:
                raise RuntimeError("RuntimeError: Cannot close a " + ("closed arc" if self.children else "beat") + ".")
            elif self.children and not self[-1].content:
                self[-1].surface(content)
            else:
                self.content = content
                self.edited = True

    @property
    def open(self) -> bool:
        """Checks if this node is an open Arc (content is None).
        
        When called on the Root, this indicates if the entire story stack is open for new events.

        Returns:
            bool: True if open, False otherwise.
        """
        return not self.content

    @property
    def can_dive(self) -> bool:
        """Checks if the node is open AND hasn't reached the maximum allowed depth.
        
        Used by the LLM Tool wrapper to dynamically mask the `dive` tool.

        Returns:
            bool: True if diving is permitted, False otherwise.
        """
        return self.open and self.depth < self.ctx.config.worldline_max_depth

    @property
    def latest(self) -> "Worldline":
        """Retrieves the deepest open Arc node currently active in the Stack.

        Returns:
            Worldline: The deepest open Arc.
        """
        with self.lock:
            if self.children and not self[-1].content:
                return self[-1].latest
            return self

    @property
    def depth(self) -> int:
        """Calculates the number of nested open Arcs below (and including) this node.
        
        When called on the Root, this effectively measures the mathematical Height 
        of the open stack, which determines if a new `dive` is permitted.

        Returns:
            int: The depth of the open stack.
        """
        with self.lock:
            if self.content:
                return 0
            elif self.children:
                return self[-1].depth + 1
            return 1

    def _get_content(self, verbosity: int = 1) -> str:
        """Packages the narrative data into a string for the LLM context window.
        
        Args:
            verbosity (int, optional): If 0, truncates children to most recent K entries and does not display depth.
                If 1, returns all children and displays depth, but children's children and beyond are limited to most recent K entries and do not display depth.
                If 2, returns all children and displays depth. Defaults to 1.
                         
        Returns:
            str: A formatted string tree of events, with a `>>>` pointer indicating 
                 the active injection point for the next event.
        """
        with self.lock:
            # open arc
            if not self.content:
                out = [f"[Worldline] {self.name} (current depth {self.depth}):" if verbosity > 0 else f"{self.name}:"]
                for uid in self.children if verbosity > 0 else self.children[-self.ctx.config.worldline_recall_k:]:
                    child = self[uid]
                    for line in child._get_content(2 if verbosity == 2 else 0).splitlines():
                        out.append("\n    ")
                        out.append(line)
                if not self.children or self[-1].content:
                    out.append("\n>>>")
                return "".join(out)
            # beat or closed arc
            return f"{self.name}: {self.content}"

    def _unpack(self, state: dict) -> None:
        """Applies a packed state dictionary to internal variables.

        Private implementation. 

        Args:
            state (dict): The packed dictionary representation.
        """
        self.name = state["name"]
        self.content = state["content"]
        self.children = state["child_UIDs"]


    def _pack(self) -> dict:
        """Returns a dictionary representation of current narrative variables.

        Private implementation. 

        Returns:
            dict: The serialized state.
        """
        return {
            "name": self.name,
            "content": self.content,
            "child_UIDs": self.children,
        }

    def __getitem__(self, idx: typing.Union[int, UID]) -> "Worldline":
        if isinstance(idx, int): 
            return self[self.children[idx]]
        return self.ctx.registry[idx]

    def _tool_beat(self, name: str, content: str) -> str:
        """DSPy tool wrapper for appending a beat to the Worldline.
        
        Args:
            name (str): A short name for the beat.
            content (str): The narrative content of the event.
            
        Returns:
            str: A success message with the new Worldline state, or an error string.
        """
        if not self.open:
            return "Error: Arc closed. No changes have been made."
        n_sentences = count_sentences(content)
        if n_sentences > self.ctx.config.worldline_max_entry_size:
            return f"Error: Content is too long ({n_sentences} sentences). Max length is {self.ctx.config.worldline_max_entry_size} sentences. No changes have been made."
        self.beat(name, content)
        return f"Success: Step appended. Worldline state: {self.get_content()}"

    @property
    def tool_beat(self) -> dspy.Tool:
        return dspy.Tool(
            self._tool_beat,
            f"{self.name} - Beat",
            "Adds a beat (leaf) entry to this Worldline. Use this for recording individual events or thoughts.",
            arg_desc={
                "name": "A short name for this beat.",
                "content": f"Content for this beat. Recommended {self.ctx.config.worldline_avg_entry_size} sentences, maximum {self.ctx.config.worldline_max_entry_size} sentences.",
            }
        )

    def _tool_dive(self, name: str) -> str:
        """DSPy tool wrapper for opening a new sub-arc.
        
        Args:
            name (str): The name of the new Arc.
            
        Returns:
            str: A success message with the new Worldline state, or an error string.
        """
        if not self.open:
            return "Error: Arc closed. No changes have been made."
        if self.depth == self.ctx.config.worldline_max_depth:
            return f"Error: Worldline already at max depth of {self.depth}. No changes have been made."
        self.dive(name)
        return f"Success: Arc diving to level {self.depth} appended. Worldline state: {self.get_content()}"


    @property
    def tool_dive(self) -> dspy.Tool:
        return dspy.Tool(
            self._tool_dive,
            f"{self.name} - Dive",
            "Adds an Arc entry to this Worldline, diving to the next level of detail.",
            arg_desc={
                "name": "A short name for this Arc, succinctly describing what it entails.",
            }
        )

    def _tool_surface(self, summary: str) -> str:
        """DSPy tool wrapper for closing and summarizing the deepest open Arc.
        
        Args:
            summary (str): The generated summary of the completed arc.
            
        Returns:
            str: A success message with the new Worldline state, or an error string.
        """
        if not self.open:
            return "Error: Arc already closed. No changes have been made."
        if self.depth == 0:
            return "Error: Cannot surface from depth 0. No changes have been made."
        n_sentences = count_sentences(summary)
        if n_sentences > self.ctx.config.worldline_max_entry_size:
            return f"Error: Summary is too long ({n_sentences} sentences). Max length is {self.ctx.config.worldline_max_entry_size} sentences. No changes have been made."
        self.surface(summary)
        return f"Success: Arc completed. Worldline state: {self.get_content()}"

    @property
    def tool_surface(self) -> dspy.Tool:
        return dspy.Tool(
            self._tool_surface,
            f"{self.name} - Surface",
            "Completes the current sub-arc, adding a summary, and surfacing to the previous depth level. Closed Arcs show only their summary and their contents will be inaccessible.",
            arg_desc={
                "summary": f"A summary of the current arc. Make sure to capture all relevant details. Recommended {self.ctx.config.worldline_avg_entry_size} sentences, maximum {self.ctx.config.worldline_max_entry_size} sentences.",
            }
        )

    @property
    def tools(self) -> list[dspy.Tool]:
        """Returns a list of DSPy tools exposed by this object."""
        return [self.tool_beat, self.tool_dive, self.tool_surface]


