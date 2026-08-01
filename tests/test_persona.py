import pytest
from unittest.mock import MagicMock, patch
import dspy

from worldline.uid import UID
from worldline.data import Context, Note, UIDGenerator, PageCounter, Config
from worldline.worldline import Worldline
from worldline.library import Library
from worldline.sketchpad import Sketchpad
from worldline.agent import Actor, WorldlineAgent
from worldline.persona import Persona, AgentPersona, Action

@pytest.fixture
def mock_ctx():
    """Provides a real global context for Pydantic validation."""
    return Context(
        uid_generator=UIDGenerator(),
        page_counter=PageCounter(),
        config=Config()
    )

@pytest.fixture
def base_actor(mock_ctx):
    """Provides an initialized Actor with its sub-notes properly registered."""
    actor = Actor(ctx=mock_ctx, name="Test Actor", uid=UID("actor-123"))
    
    timeline = Worldline(ctx=mock_ctx, name="Test Timeline", uid=actor.timeline_uid)
    memory = Library(ctx=mock_ctx, name="Test Memory", uid=actor.memory_uid)
    moment = Sketchpad(ctx=mock_ctx, name="Test Moment", uid=actor.moment_uid)
    
    mock_ctx.registry[actor.timeline_uid] = timeline
    mock_ctx.registry[actor.memory_uid] = memory
    mock_ctx.registry[actor.moment_uid] = moment
    
    return actor

class TestPersona:
    """Tests for the base Persona class."""
    def test_sys_scene_notes_formatting(self, base_actor):
        persona = Persona(ctx=base_actor.ctx, name="Hero", uid=base_actor.uid)
        assert "Hero" in persona.sys_scene_notes

    def test_turn_default(self, base_actor):
        persona = Persona(ctx=base_actor.ctx, name="Hero", uid=base_actor.uid)
        action = persona.turn(scene="A goblin attacks.")
        assert action.initiative == 0
        assert action.action is None

class TestAgentPersona:
    """Tests for the AgentPersona game loop and DSPy integration."""
    def test_lazy_agent_instantiation(self, base_actor):
        # Convert base_actor to AgentPersona manually for testing
        persona = AgentPersona(ctx=base_actor.ctx, name="Hero", uid=base_actor.uid)
        persona.timeline_uid = base_actor.timeline_uid
        persona.memory_uid = base_actor.memory_uid
        persona.moment_uid = base_actor.moment_uid
        
        assert persona._agent is None
        agent = persona.agent
        assert isinstance(agent, WorldlineAgent)
        assert persona._agent is not None
        assert persona.agent is agent  # Check singleton caching

    @patch.object(AgentPersona, 'agent')
    def test_turn_valid_action(self, mock_agent_property, base_actor):
        persona = AgentPersona(ctx=base_actor.ctx, name="Hero", uid=base_actor.uid)
        
        # Mock the DSPy Prediction return object
        mock_prediction = MagicMock()
        mock_prediction.initiative = 8
        mock_prediction.action = "I draw my sword."
        
        # The agent property is a MagicMock, so calling it returns the prediction
        mock_agent_property.return_value = mock_prediction
        
        action = persona.turn(scene="A goblin attacks.")
        
        assert action.initiative == 8
        assert action.action == "I draw my sword."

    @patch.object(AgentPersona, 'agent')
    def test_turn_zero_initiative(self, mock_agent_property, base_actor):
        persona = AgentPersona(ctx=base_actor.ctx, name="Hero", uid=base_actor.uid)
        
        # Mock an initiative of 0 (decides not to act)
        mock_prediction = MagicMock()
        mock_prediction.initiative = 0
        mock_prediction.action = "N/A"
        
        mock_agent_property.return_value = mock_prediction
        action = persona.turn(scene="Nothing is happening.")
        
        assert action.initiative == 0
        assert action.action is None

    @patch.object(AgentPersona, 'agent')
    def test_turn_exception_handling(self, mock_agent_property, base_actor):
        persona = AgentPersona(ctx=base_actor.ctx, name="Hero", uid=base_actor.uid)
        
        # Mock DSPy failing to parse output and throwing an error
        mock_agent_property.side_effect = Exception("DSPy Parse Error")
        
        # The turn should safely catch the exception and return no action
        action = persona.turn(scene="A goblin attacks.")
        
        assert action.initiative == 0
        assert action.action is None
