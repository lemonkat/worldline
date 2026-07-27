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

from worldline.data import Note

class Worldline(Note):
    """A recursive narrative container representing the active story stack.

    A Worldline acts as either a Beat (a leaf node representing a single action with raw content) 
    or an Arc (a parent frame grouping multiple children, where 'content' acts as the summary of those children).
    
    The tree implements an automatic routing architecture: operations called on the Root 
    are mathematically guaranteed to cascade down to the deepest open Arc.

    Attributes:
        title (str): The title of the Arc or Beat.
        content (str | None): The raw text or summary content. None if the Arc is open.
        children (list[Worldline]): The ordered list of nested Worldlines.
    """
    title: str = "ROOT"
    content: typing.Optional[str] = None
    children: list["Worldline"] = Field(default_factory=list)

    def step(self, title: str, content: str) -> None:
        """Adds a single, atomic Beat node to the deepest open Arc.
        
        Args:
            title (str): A short description of the action.
            content (str): The raw narrative text of the event.
            
        Raises:
            RuntimeError: If the Worldline (or the deepest Arc) is already closed.
        """
        if not self.open:
            raise RuntimeError("RuntimeError: Cannot append to a " + ("closed arc" if self.children else "beat") + ".")
        if self.children and not self.children[-1].content:
            self.children[-1].step(title, content)
        else:
            self.children.append(Worldline(ctx=self.ctx, title=title, content=content))
            self.edited = True

    def dive(self, title: str) -> None:
        """Opens a new nested Arc within the deepest open Arc.
        
        Args:
            title (str): The title of the new Arc.
            
        Raises:
            RuntimeError: If the Worldline (or the deepest Arc) is already closed.
        
        Warns:
            RuntimeWarning: If opening the new Arc would exceed worldline_max_depth.
        """
        if not self.open:
            raise RuntimeError("RuntimeError: Cannot append to a " + ("closed arc" if self.children else "beat") + ".")
        
        if self.depth >= self.ctx.config.worldline_max_depth:
            warnings.warn(f"Diving past maximum depth of {self.ctx.config.worldline_max_depth}.", RuntimeWarning)
        
        if self.children and not self.children[-1].content:
            self.children[-1].dive(title)
        else:
            self.children.append(Worldline(ctx=self.ctx, title=title))
            self.edited = True

    def surface(self, content: str) -> None:
        """Closes the deepest open Arc by populating its summary content.
        
        Args:
            content (str): The LLM-generated summary of all events that occurred inside the Arc.
            
        Raises:
            RuntimeError: If the Worldline is already closed, meaning there are no open Arcs left.
        """
        if not self.open:
            raise RuntimeError("RuntimeError: Cannot close a " + ("closed arc" if self.children else "beat") + ".")
        elif self.children and not self.children[-1].content:
            self.children[-1].surface(content)
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
    def latest(self) -> typing.Self:
        """Retrieves the deepest open Arc node currently active in the Stack.

        Returns:
            typing.Self: The deepest open Arc.
        """
        if self.children and not self.children[-1].content:
            return self.children[-1].latest
        return self

    @property
    def depth(self) -> int:
        """Calculates the number of nested open Arcs below (and including) this node.
        
        When called on the Root, this effectively measures the mathematical Height 
        of the open stack, which determines if a new `dive` is permitted.

        Returns:
            int: The depth of the open stack.
        """
        if self.content:
            return 0
        elif self.children:
            return self.children[-1].depth + 1
        return 1

    def _get_content(self, full: bool = True) -> str:
        """Packages the narrative data into a string for the LLM context window.
        
        Args:
            full (bool, optional): If True, returns all children (used for the Top Level).
                         If False, truncates children to the most recent K entries. Defaults to True.
                         
        Returns:
            str: A formatted string tree of events, with a `>>>` pointer indicating 
                 the active injection point for the next event.
        """
        # open arc
        if not self.content:
            out = [self.title, ":"]
            for child in self.children if full else self.children[-self.ctx.config.worldline_recall_k:]:
                for line in child._get_content(full=False).splitlines():
                    out.append("\n    ")
                    out.append(line)
            if not self.children or self.children[-1].content:
                out.append("\n>>>")
            return "".join(out)
        # beat or closed arc
        return f"{self.title}: {self.content}"

    def _unpack(self, state: dict) -> None:
        """Applies a packed state dictionary to internal variables.

        Private implementation. 

        Args:
            state (dict): The packed dictionary representation.
        """
        self.title = state["title"]
        self.content = state["content"]
        self.children = [self.ctx.registry[uid] for uid in state["child_UIDs"]]


    def _pack(self) -> dict:
        """Returns a dictionary representation of current narrative variables.

        Private implementation. 

        Returns:
            dict: The serialized state.
        """
        return {
            "title": self.title,
            "content": self.content,
            "child_UIDs": [child.uid for child in self.children],
        }
