import random
import typing
import dspy

from pydantic import model_validator

from worldline.uid import UID
from worldline.data import Context, Note, lock_notes
from worldline.worldline import Worldline
from worldline.library import Library
from worldline.sketchpad import Sketchpad

def _tool_d20(action: str, threshold: int, outcome_low: str, outcome_high: str) -> str:
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
    """
    def __init__(
        self, 
        ctx: Context,
        signature: type[dspy.Signature], 
        notes: list[Note], 
        tools: typing.Optional[list[dspy.Tool]] = None,
        max_iters: int = 30,
        use_react_v2: bool = False,
    ) -> None:
        super().__init__()
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


class Actor(Note):
    """Base class for any autonomous entity with agency in the narrative.

    An Actor is a composite Note that maintains its own personal timeline (Worldline), 
    knowledge base (Library), and working memory (Sketchpad). It automatically 
    manages the lifecycle, locking, and save/load (pack/unpack) serialization 
    of its sub-Notes.

    Attributes:
        name (str): The name of the Actor.
        timeline_uid (UID, optional): Pointer to the Actor's Worldline (recent events).
        memory_uid (UID, optional): Pointer to the Actor's Library (world knowledge/beliefs).
        moment_uid (UID, optional): Pointer to the Actor's Sketchpad (current thoughts).
    """
    name: str = "Unnamed Actor"
    timeline_uid: typing.Optional[UID] = None
    memory_uid: typing.Optional[UID] = None
    moment_uid: typing.Optional[UID] = None

    sys_name: typing.ClassVar[str] = "Actor"
    sys_desc: typing.ClassVar[str] = "Base class. Not meant to be used directly. IF YOU SEE THIS SOMETHING HAS GONE WRONG."

    @model_validator(mode="after")
    def _setup_actor(self) -> "Actor":
        """Post-initialization validation hook for Actor sub-Notes."""
        if self.timeline_uid is None:
            self.timeline_uid = Worldline(ctx=self.ctx, name=f"{self.name} - Timeline").uid

        if self.memory_uid is None:
            self.memory_uid = Library(ctx=self.ctx, name=f"{self.name} - Memory").uid

        if self.moment_uid is None:
            self.moment_uid = Sketchpad(ctx=self.ctx, name=f"{self.name} - Moment").uid
                    
        return self

    def _pack(self) -> dict:
        """Packages the Actor's sub-Note UIDs into a state snapshot for saving."""
        return {
            "data_UIDs": [self.timeline_uid, self.memory_uid, self.moment_uid],
        }

    def _unpack(self, state: dict) -> None:
        """Restores the Actor's sub-Note UIDs from a state snapshot."""
        self.timeline_uid, self.memory_uid, self.moment_uid = state["data_UIDs"]

    @property
    def timeline(self) -> typing.Optional[Worldline]:
        """Returns the Actor's Worldline instance (recent subjective events)."""
        return typing.cast(Worldline, self.ctx.registry[self.timeline_uid]) if self.timeline_uid else None

    @property
    def memory(self) -> typing.Optional[Library]:
        """Returns the Actor's Library instance (long-term knowledge and beliefs)."""
        return typing.cast(Library, self.ctx.registry[self.memory_uid]) if self.memory_uid else None

    @property
    def moment(self) -> typing.Optional[Sketchpad]:
        """Returns the Actor's Sketchpad instance (internal monologue and current plans)."""
        return typing.cast(Sketchpad, self.ctx.registry[self.moment_uid]) if self.moment_uid else None

    @property
    def tools(self) -> list[dspy.Tool]:
        tools = [tool_d20]
        tools.extend(self.timeline.tools)
        tools.extend(self.memory.tools)
        tools.extend(self.moment.tools)
        return tools

    # for when you dont want to edit timeline or moment, for long term only things
    @property
    def lookup_tools(self) -> list[dspy.Tool]:
        tools = [tool_d20]
        tools.extend(self.memory.tools)
        return tools
        