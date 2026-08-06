import pytest
from unittest.mock import MagicMock, patch

from worldline.actor import Actor, Directive
from worldline.data import Context, Config, PageCounter
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

def test_directive_initialization(mock_ctx):
    directive = Directive(ctx=mock_ctx)
    assert directive.instructions == "N/A"
    assert "N/A" in directive._get_content()

def test_directive_setter(mock_ctx):
    directive = Directive(ctx=mock_ctx)
    directive.edited = False
    
    directive.instructions = "Write in the style of Shakespeare."
    assert directive.instructions == "Write in the style of Shakespeare."
    assert directive.edited is True
    
    directive.edited = False
    directive.instructions = "Write in the style of Shakespeare."
    assert directive.edited is False

def test_directive_pack_unpack(mock_ctx):
    directive = Directive(ctx=mock_ctx)
    directive.instructions = "Grimdark fantasy."
    
    state = directive.pack()
    
    directive2 = Directive(ctx=mock_ctx)
    directive2.unpack(state)
    assert directive2.instructions == "Grimdark fantasy."

def test_actor_initialization(mock_ctx):
    actor = Actor(ctx=mock_ctx, name="Test Actor")
    
    assert actor.name == "Test Actor"
    assert actor.directive is not None
    assert isinstance(actor.directive, Directive)
    assert actor.timeline is not None
    assert isinstance(actor.timeline, Worldline)
    assert actor.memory is not None
    assert isinstance(actor.memory, Library)
    assert actor.moment is not None
    assert isinstance(actor.moment, Sketchpad)
    
    # Check that they were registered in the context registry
    assert mock_ctx.registry[actor.directive_uid] == actor.directive
    assert mock_ctx.registry[actor.timeline_uid] == actor.timeline
    assert mock_ctx.registry[actor.memory_uid] == actor.memory
    assert mock_ctx.registry[actor.moment_uid] == actor.moment

def test_actor_setup_bypassed_if_loading(mock_ctx):
    mock_ctx.is_loading = True
    actor = Actor(ctx=mock_ctx, name="Loading Actor")
    
    assert actor.directive_uid is None
    assert actor.timeline_uid is None
    assert actor.memory_uid is None
    assert actor.moment_uid is None

def test_actor_properties_resolve_from_registry(mock_ctx):
    actor = Actor(ctx=mock_ctx, name="Properties Actor")
    
    assert actor.directive == mock_ctx.registry[actor.directive_uid]
    assert actor.timeline == mock_ctx.registry[actor.timeline_uid]
    assert actor.memory == mock_ctx.registry[actor.memory_uid]
    assert actor.moment == mock_ctx.registry[actor.moment_uid]

def test_actor_pack_unpack(mock_ctx):
    actor = Actor(ctx=mock_ctx, name="Pack Actor")
    state = actor.pack()
    
    # Create a new actor and unpack into it
    actor2 = Actor(ctx=mock_ctx, name="Dummy")
    actor2.unpack(state)
    
    assert actor2.directive_uid == actor.directive_uid
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

def test_actor_tool_lore(mock_ctx):
    actor = Actor(ctx=mock_ctx, name="Lore Actor")
    
    # Mock the lore_agent call to avoid running DSPy ReAct
    with patch("worldline.actor.WorldlineAgent.__call__") as mock_agent_call:
        mock_agent_call.return_value = MagicMock(response="Mocked Lore Response")
        
        # Test tool_lore execution
        result = actor._tool_lore("Who is the king?")
        
        assert result == "Mocked Lore Response"
        mock_agent_call.assert_called_once_with(query="Who is the king?")
        
        # Verify tool_lore properties
        tool = actor.tool_lore
        assert tool.name == "Lore lookup"
        
        # Verify lore_agent initialization
        lore_agent = actor.lore_agent
        assert actor.directive in lore_agent.notes
        assert actor.timeline in lore_agent.notes
        assert actor.memory in lore_agent.notes
        assert actor.moment in lore_agent.notes
        assert lore_agent.tools == actor.memory.tools
