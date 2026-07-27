"""
[Phase 0.5: Sketchpads]

Minimal Note objects w/ title, editable content for simple tasks.
"""

import dspy

from worldline.data import Note

class Sketchpad(Note):
    """A minimal, editable Note object acting as a persistent text field.
    
    Provides a simple interface for the ReAct agent to jot down ongoing plans 
    or temporary scratchpad thoughts that persist across turns.
    
    Attributes:
        title (str): The title of the Sketchpad.
        content (str): The editable text content.
    """
    title: str = "Untitled"
    content: str = ""

    def _get_content(self) -> str:
        return f"[Sketchpad] {self.title}: {self.content}"

    def _unpack(self, state: dict) -> None:
        """Applies a packed state dictionary to internal variables.

        Private implementation. 

        Args:
            state (dict): The packed dictionary representation.
        """
        self.title = state["title"]
        self.content = state["content"]


    def _pack(self) -> dict:
        """Returns a dictionary representation of current narrative variables.

        Private implementation. 

        Returns:
            dict: The serialized state.
        """
        return {
            "title": self.title,
            "content": self.content,
        }

    @property
    def tools(self) -> list[dspy.Tool]:
        """Returns a list of DSPy tools exposed by this object."""
        return [self.tool_write]

    def _tool_write(self, content: str) -> str:
        """DSPy tool wrapper for overwriting the sketchpad's content.
        
        Args:
            content (str): The new text to write into the sketchpad.
            
        Returns:
            str: A success message indicating the content was updated.
        """
        self.content = content
        self.edited = True
        return "Content updated successfully."

    @property
    def tool_write(self) -> dspy.Tool:
        return dspy.Tool(
            self._tool_write,
            f"{self.title} - Write",
            "Sets the content of this Sketchpad.",
        )