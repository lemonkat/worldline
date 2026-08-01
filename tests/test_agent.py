import pytest
import dspy
import asyncio
import threading
from unittest.mock import MagicMock, patch

from worldline.agent import _tool_d20, tool_d20, WorldlineAgent, Actor
from worldline.data import Context, Config, Note, PageCounter
from worldline.uid import UIDGenerator
from worldline.worldline import Worldline
from worldline.library import Library
from worldline.sketchpad import Sketchpad

@pytest.fixture
def mock_ctx():
    return Context(
        uid_generator=UIDGenerator(),
        page_counter=PageCounter(),
        config=Config()
    )

class MockNote(Note):
    name: str = "Mock"
    content: str = ""
    
    sys_name = "MockNote"
    sys_desc = "A mock note for testing."
    
    def _get_content(self) -> str:
        return self.content

    @property
    def tools(self) -> list[dspy.Tool]:
        return [dspy.Tool(lambda: "", "Mock Tool", "")]


def test_tool_d20():
    # Test bounds
    assert "out of bounds" in _tool_d20("attack", 0, "miss", "hit")
    assert "out of bounds" in _tool_d20("attack", 21, "miss", "hit")
    
    # Test deterministic high
    assert "(HIGH)" in _tool_d20("attack", 1, "miss", "hit")

    # Test deterministic low
    with patch("worldline.agent.random.randint", return_value=5):
        assert "(LOW)" in _tool_d20("attack", 10, "miss", "hit")


def test_agent_initialization(mock_ctx):
    n1 = MockNote(ctx=mock_ctx)
    n2 = MockNote(ctx=mock_ctx)
    
    class TestSig(dspy.Signature):
        task = dspy.InputField()
        response = dspy.OutputField()

    agent = WorldlineAgent(
        ctx=mock_ctx,
        signature=TestSig,
        notes=[n1, n2],
        tools=n1.tools
    )
    
    assert len(agent.mutable_notes) == 2
    assert agent.mutable_notes[0].uid < agent.mutable_notes[1].uid
    assert "MockNote" in agent.signature.instructions


def test_agent_forward_locks(mock_ctx):
    n1 = MockNote(ctx=mock_ctx)
    n2 = MockNote(ctx=mock_ctx)
    
    n1.lock = MagicMock()
    n2.lock = MagicMock()

    class TestSig(dspy.Signature):
        task = dspy.InputField()
        response = dspy.OutputField()

    agent = WorldlineAgent(
        ctx=mock_ctx,
        signature=TestSig,
        notes=[n1, n2],
        tools=n1.tools
    )

    agent.react = MagicMock()
    agent.react.return_value = MagicMock(response="Mocked")

    agent.forward(task="Test")
    
    n1.lock.__enter__.assert_called_once()
    n1.lock.__exit__.assert_called_once()
    n2.lock.__enter__.assert_called_once()
    n2.lock.__exit__.assert_called_once()


@pytest.mark.asyncio
async def test_agent_aforward_locks(mock_ctx):
    n1 = MockNote(ctx=mock_ctx)
    
    n1.lock = MagicMock()

    class TestSig(dspy.Signature):
        task = dspy.InputField()
        response = dspy.OutputField()

    agent = WorldlineAgent(
        ctx=mock_ctx,
        signature=TestSig,
        notes=[n1],
        tools=n1.tools
    )

    async def mock_aforward(**kwargs):
        return MagicMock(response="Mocked")

    agent.react = MagicMock()
    agent.react.aforward = mock_aforward

    await agent.aforward(task="Test")
    
    n1.lock.__enter__.assert_called_once()
    n1.lock.__exit__.assert_called_once()

def test_actor_initialization(mock_ctx):
    actor = Actor(ctx=mock_ctx, name="Test Actor")
    
    assert actor.name == "Test Actor"
    assert actor.timeline is not None
    assert isinstance(actor.timeline, Worldline)
    assert actor.memory is not None
    assert isinstance(actor.memory, Library)
    assert actor.moment is not None
    assert isinstance(actor.moment, Sketchpad)
    
    # Check that they were registered in the context registry
    assert mock_ctx.registry[actor.timeline_uid] == actor.timeline
    assert mock_ctx.registry[actor.memory_uid] == actor.memory
    assert mock_ctx.registry[actor.moment_uid] == actor.moment

def test_actor_setup_bypassed_if_loading(mock_ctx):
    mock_ctx.is_loading = True
    actor = Actor(ctx=mock_ctx, name="Loading Actor")
    
    assert actor.timeline_uid is None
    assert actor.memory_uid is None
    assert actor.moment_uid is None

def test_actor_properties_resolve_from_registry(mock_ctx):
    actor = Actor(ctx=mock_ctx, name="Properties Actor")
    
    assert actor.timeline == mock_ctx.registry[actor.timeline_uid]
    assert actor.memory == mock_ctx.registry[actor.memory_uid]
    assert actor.moment == mock_ctx.registry[actor.moment_uid]

def test_actor_pack_unpack(mock_ctx):
    actor = Actor(ctx=mock_ctx, name="Pack Actor")
    state = actor.pack()
    
    # Create a new actor and unpack into it
    actor2 = Actor(ctx=mock_ctx, name="Dummy")
    actor2.unpack(state)
    
    assert actor2.timeline_uid == actor.timeline_uid
    assert actor2.memory_uid == actor.memory_uid
    assert actor2.moment_uid == actor.moment_uid

def test_actor_tools(mock_ctx):
    actor = Actor(ctx=mock_ctx, name="Tool Actor")
    
    tools = actor.tools
    tool_names = [t.name for t in tools]
    
    # Should contain tool_d20
    assert "D20" in tool_names
    
    lookup_tools = actor.lookup_tools
    lookup_tool_names = [t.name for t in lookup_tools]
    
    # Lookup tools should not contain moment/timeline tools
    assert "D20" not in lookup_tool_names
    assert len(lookup_tools) < len(tools)
