"""
[Phase Two: Actor/Engine]

Base Actor infrastructure and Directive instructions for Project Worldline.

This module defines the foundational Actor class, representing entities that
possess narrative agency, memory, and timelines. It also introduces the
Directive class for injecting global storytelling instructions.
"""

import typing

import dspy
from pydantic import model_validator

from worldline.uid import UID
from worldline.data import Note
from worldline.worldline import Worldline
from worldline.library import Library
from worldline.sketchpad import Sketchpad
from worldline.agent import tool_d20, WorldlineAgent

class Directive(Note):
    """Read-only instructions for story contents.
    
    Attributes:
        instructions (str): instructions for any agents involved in the writing process.
    """

    sys_name: typing.ClassVar[str] = "Directive"
    sys_desc: typing.ClassVar[str] = "A read-only text field containing miscellaneous storytelling instructions."

    _instructions: str = "N/A"

    def _pack(self) -> dict:
        """Packages the Directive's current state into a dictionary.

        Returns:
            dict: The serialized state.
        """
        return {"instructions": self._instructions}

    def _unpack(self, state: dict) -> None:
        """Restores the Directive's state from a packed dictionary.

        Args:
            state (dict): The packed dictionary representation.
        """
        self._instructions = state["instructions"]

    @property
    def instructions(self) -> str:
        """Returns the current instructions."""
        with self.lock:
            return self._instructions

    @instructions.setter
    def instructions(self, instructions: str) -> None:
        """Sets the instructions, marking the Note as edited if changed."""
        with self.lock:
            if instructions != self._instructions:
                self._instructions = instructions
                self.edited = True

    def _get_content(self) -> str:
        """Returns the narrative text of this Directive for LLM contexts.

        Returns:
            str: The formatted narrative string.
        """
        return f"[Directive] Additional storytelling instructions: {self._instructions}"

class Actor(Note):
    """Base class for any autonomous entity with agency in the narrative.

    An Actor is a composite Note that maintains its own personal timeline (Worldline), 
    knowledge base (Library), and working memory (Sketchpad). It automatically 
    manages the lifecycle, locking, and save/load (pack/unpack) serialization 
    of its sub-Notes.

    Attributes:
        name (str): The name of the Actor.
        directive_uid (UID, optional): Pointer to the Actor's Directive (miscellaneous storytelling instructions).
        timeline_uid (UID, optional): Pointer to the Actor's Worldline (recent events).
        memory_uid (UID, optional): Pointer to the Actor's Library (world knowledge/beliefs).
        moment_uid (UID, optional): Pointer to the Actor's Sketchpad (current thoughts).
        _lore_agent (typing.Optional[WorldlineAgent]): Cached singleton of the nested ReAct agent used for lore lookups.
        LORE_AGENT_SIGNATURE (typing.ClassVar[type[dspy.Signature]]): Specialized LLM formatting instructions for the lore agent.
    """
    name: str = "Unnamed Actor"
    directive_uid: typing.Optional[UID] = None
    timeline_uid: typing.Optional[UID] = None
    memory_uid: typing.Optional[UID] = None
    moment_uid: typing.Optional[UID] = None

    sys_name: typing.ClassVar[str] = "Actor"
    sys_desc: typing.ClassVar[str] = "Base class. Not meant to be used directly. IF YOU SEE THIS SOMETHING HAS GONE WRONG."

    _lore_agent: typing.Optional[WorldlineAgent] = None

    LORE_AGENT_SIGNATURE: typing.ClassVar[type[dspy.Signature]] = dspy.make_signature(
        signature={
            "query": (
                str, 
                dspy.InputField(desc="What someone wants to know."),
            ),
            "response": (
                str, 
                dspy.OutputField(desc="Your response, either found or created story lore."),
            ),
        },
        instructions="""Answer the provided query about the current state and lore of the story so the requester can continue it.
Find the information if it exists, and ONLY if it does not exist, come up with what it should be and save it for future reference.
Provide plenty of information, too much is better than too little here.""",
    )

    @model_validator(mode="after")
    def _setup_actor(self) -> typing.Self:
        """Post-initialization validation hook for Actor sub-Notes."""
        if not self.ctx.is_loading:
            if self.directive_uid is None:
                self.directive_uid = Directive(ctx=self.ctx).uid

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
            "directive_UID": self.directive_uid,
            "timeline_UID": self.timeline_uid,
            "memory_UID": self.memory_uid,
            "moment_UID": self.moment_uid,
        }

    def _unpack(self, state: dict) -> None:
        """Restores the Actor's sub-Note UIDs from a state snapshot."""
        self.directive_uid = state["directive_UID"]
        self.timeline_uid = state["timeline_UID"]
        self.memory_uid = state["memory_UID"]
        self.moment_uid = state["moment_UID"]

    @property
    def directive(self) -> typing.Optional[Directive]:
        """Returns the Actor's Directive instance (miscellaneous storytelling instructions)."""
        return typing.cast(Directive, self.ctx.registry[self.directive_uid]) if self.directive_uid else None

    @directive.setter
    def directive(self, directive: typing.Optional[Directive] = None) -> None:
        self.directive_uid = None if directive is None else directive.uid

    @property
    def timeline(self) -> typing.Optional[Worldline]:
        """Returns the Actor's Worldline instance (recent subjective events)."""
        return typing.cast(Worldline, self.ctx.registry[self.timeline_uid]) if self.timeline_uid else None

    @timeline.setter
    def timeline(self, timeline: typing.Optional[Worldline] = None) -> None:
        self.timeline_uid = None if timeline is None else timeline.uid

    @property
    def memory(self) -> typing.Optional[Library]:
        """Returns the Actor's Library instance (long-term knowledge and beliefs)."""
        return typing.cast(Library, self.ctx.registry[self.memory_uid]) if self.memory_uid else None

    @memory.setter
    def memory(self, memory: typing.Optional[Library] = None) -> None:
        self.memory_uid = None if memory is None else memory.uid

    @property
    def moment(self) -> typing.Optional[Sketchpad]:
        """Returns the Actor's Sketchpad instance (internal monologue and current plans)."""
        return typing.cast(Sketchpad, self.ctx.registry[self.moment_uid]) if self.moment_uid else None

    @moment.setter
    def moment(self, moment: typing.Optional[Sketchpad] = None) -> None:
        self.moment_uid = None if moment is None else moment.uid

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
        return self.memory.tools

    @property
    def lore_agent(self) -> WorldlineAgent:
        """Lazily instantiates and returns the nested WorldlineAgent for lore lookups.

        Constructs the DSPy signature and ReAct module only upon first access,
        ensuring safe instantiation when loading from save files. This nested agent 
        has access to the Actor's memory tools to search or create records.

        Returns:
            WorldlineAgent: The configured DSPy ReAct agent for lore lookups.
        """
        if self._lore_agent is None:
            self._lore_agent = WorldlineAgent(
                self.ctx,
                self.LORE_AGENT_SIGNATURE,
                [self.directive, self.timeline, self.memory, self.moment],
                self.memory.tools,
            )

        return self._lore_agent

    def _tool_lore(self, query: str) -> str:
        """DSPy tool wrapper that executes the nested Lore Agent to answer a query.
        
        Args:
            query (str): The information the caller is looking for.
            
        Returns:
            str: The response from the Lore Agent, containing found or invented lore.
        """
        return self.lore_agent(query=query).response

    @property
    def tool_lore(self) -> dspy.Tool:
        """Returns the DSPy Tool object for lore lookups.
        
        This tool delegates queries to a nested ReAct agent with access to this Actor's 
        memory tools, allowing it to search and invent records without exposing raw 
        database operations to the caller.
        
        Returns:
            dspy.Tool: The configured lore lookup tool.
        """
        return dspy.Tool(
            self._tool_lore,
            "Lore lookup",
            """Use this tool to find information you can't find elsewhere, but use it only when necessary.
If there is information that should exist but you don't have it, DO NOT make it up - request it using this tool to ensure consistency.
Make sure to record in your Memory any response this tool returns so you have it for future use.""",
            arg_desc={
                "query": "The information you are looking for. If possible, mention what you do know relating to your query to help the system.",
            }
        )
        