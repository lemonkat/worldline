"""
[Phase 0.5: Sketchpads]

Minimal Note objects w/ name, editable content for simple tasks.
"""

import typing

import dspy

from worldline.data import Note

class Sketchpad(Note):
    """A minimal, editable Note object acting as a persistent text field.
    
    Provides a simple interface for the ReAct agent to jot down ongoing plans 
    or temporary scratchpad thoughts that persist across turns.
    
    Attributes:
        name (str): The name of the Sketchpad.
        content (str): The editable text content.
    """
    name: str = "Unnamed"
    content: str = ""

    sys_name: typing.ClassVar[str] = "Sketchpad"
    sys_desc: typing.ClassVar[str] = "A temporary text field that persists across turns. Use the write tool to store notes, reminders, or intermediate plans."

    def _get_content(self) -> str:
        return f"[Sketchpad] {self.name}: {self.content}"

    def _unpack(self, state: dict) -> None:
        """Applies a packed state dictionary to internal variables.

        Private implementation. 

        Args:
            state (dict): The packed dictionary representation.
        """
        self.name = state["name"]
        self.content = state["content"]


    def _pack(self) -> dict:
        """Returns a dictionary representation of current narrative variables.

        Private implementation. 

        Returns:
            dict: The serialized state.
        """
        return {
            "name": self.name,
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
            f"{self.name} - Write",
            "Sets the content of this Sketchpad.",
        )