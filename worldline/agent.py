"""
[Phase 2.5: Agents and Actors]

WorldlineAgent module integrating Note system with DSPy's ReAct implementation.
"""

import random
import typing

import dspy
from dspy.utils import BaseCallback

from worldline.data import Context, Note, lock_notes

def _tool_d20(action: str, threshold: int, outcome_low: str, outcome_high: str) -> str:
    """DSPy tool wrapper for rolling a 20-sided die to determine probabilistic outcomes."""
    if threshold <= 0 or threshold > 20:
        return f"Error: threshold ({threshold}) out of bounds."
    roll = random.randint(1, 20)
    
    if roll < threshold:
        return f"Rolled {roll} (LOW). Result: {outcome_low}"
    else:
        return f"Rolled {roll} (HIGH). Result: {outcome_high}"

tool_d20 = dspy.Tool(
    _tool_d20,
    "D20",
    "Rolls a 20-sided die to determine the outcome of a random event. Use this for random choices, or risky/probabilistic events.",
    arg_desc={
        "action": "The random event.",
        "threshold": "An integer from 1 to 20 representing how high the roll must be for the high outcome to occur.",
        "outcome_low": "The first outcome. This will occur if roll < threshold.",
        "outcome_high": "The second outcome. This will occur if roll >= threshold.",
    }
)

class WorldlineAgent(dspy.Module):
    """A generic, state-aware agent powered by DSPy ReAct.

    The WorldlineAgent manages a collection of `Note` objects, dynamically 
    providing their contents to the LLM and exposing their respective tools. 
    It leverages pessimistic locking to ensure thread-safe execution across 
    concurrent multi-agent scenarios.

    Attributes:
        ctx (Context): The global engine context.
        notes (list[Note]): The Notes this agent has read/write access to.
        tools (list[dspy.Tool], optional): Additional tools to expose.
        signature (dspy.Signature): The compiled ReAct signature.
        mutable_notes (list[Note]): The subset of Notes that expose tools.
        callbacks (list[BaseCallback], optional): Callbacks to pass to dspy.Module.__init__()
    """
    def __init__(
        self, 
        ctx: Context,
        signature: type[dspy.Signature], 
        notes: list[Note], 
        tools: typing.Optional[list[dspy.Tool]] = None,
        max_iters: int = 30,
        use_react_v2: bool = False,
        callbacks: typing.Optional[list[BaseCallback]] = None,
    ) -> None:
        super().__init__(callbacks)
        self.ctx = ctx

        self.notes = notes
        self.tools = tools

        # the notes for which we have tool access, sorted by UID
        tool_names = {tool.name for tool in self.tools}
        self.mutable_notes = [note for note in self.notes if any(tool.name in tool_names for tool in note.tools)]

        self.signature = signature.prepend(
            "context", 
            dspy.InputField(desc="Initial useful information."), 
            str,
        ).append_instructions(self._build_note_desc())

        if not self.tools:
            self.react = dspy.ChainOfThought(signature=signature)
        else:
            react_class = dspy.ReActV2 if use_react_v2 else dspy.ReAct
            self.react = react_class(signature=self.signature, tools=self.tools, max_iters=max_iters)

        
    def _build_note_desc(self) -> str:
        """Dynamically constructs instructions based on active Note types.

        Iterates through the distinct types of provided Notes and concatenates 
        their `sys_name` and `sys_desc` to instruct the LLM on how to use them.

        Returns:
            str: The formatted system instructions string.
        """
        result = []
        for note_type in {type(note) for note in self.notes}:
            result.append(f"Details for {note_type.sys_name}: {note_type.sys_desc}")
        return "\n\n".join(result)

    def _build_context(self) -> str:
        """Assembles the initial context block from all active Notes.

        Calls the `initialize` method on each Note (which includes formatting 
        and injecting previous context) to build the comprehensive world state.

        Returns:
            str: The fully assembled context string.
        """
        context = ""
        for note in self.notes:
            context += ("\n\n" if context else "") + note.initialize(context)
        return context

    def forward(self, **kwargs: typing.Any) -> typing.Any:
        """Executes the ReAct loop synchronously.

        Acquires locks on all mutable Notes (in UID order to prevent deadlocks) 
        before executing the LLM loop to ensure thread-safe read/write operations.
        """
        with lock_notes(self.mutable_notes):
            return self.react(context=self._build_context(), **kwargs)

    async def aforward(self, **kwargs: typing.Any) -> typing.Any:
        """Executes the ReAct loop asynchronously.

        Safely offloads the synchronous DSPy loop to a background thread pool 
        (or uses native aforward if available) to prevent blocking the asyncio 
        event loop, while maintaining perfect thread safety via ordered locks.
        """
        with lock_notes(self.mutable_notes):
            if hasattr(self.react, "aforward"):
                return await self.react.aforward(context=self._build_context(), **kwargs)
            else:
                import asyncio
                return await asyncio.to_thread(self.react, context=self._build_context(), **kwargs)


