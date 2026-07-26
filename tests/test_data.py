import pytest
from unittest.mock import patch
import operator

from worldline.data import PageCounter, Note

def test_page_counter_init_default():
    pc = PageCounter()
    assert pc.page == 0

def test_page_counter_init_custom():
    pc = PageCounter(10)
    assert pc.page == 10

def test_page_counter_step():
    pc = PageCounter(0)
    assert pc.step() == 1
    assert pc.page == 1
    assert pc.step() == 2

def test_page_counter_str_repr():
    pc = PageCounter(5)
    assert str(pc) == "Page 5"
    assert repr(pc) == "PageCounter(5)"

def test_page_counter_int():
    pc = PageCounter(7)
    assert int(pc) == 7
    assert operator.index(pc) == 7

import numpy as np
from worldline.data import PageCounter, Note, Context, Config
from worldline.uid import UIDGenerator

# --- Note Tests ---

# Create a concrete subclass for testing the abstract Note
class MockNote(Note):
    def __init__(self, context, text: str, **kwargs):
        super().__init__(context=context, **kwargs)
        self.text = text
        
    def _get_content(self):
        return self.text
        
    def _pack(self):
        return self.text
        
    def _unpack(self, state):
        self.text = state

@pytest.fixture
def mock_ctx():
    uid_gen = UIDGenerator(0)
    pc = PageCounter(0)
    return Context(uid_generator=uid_gen, page_counter=pc, config=Config())

def test_note_lazy_evaluation(mock_ctx):
    with patch("worldline.llm.get_emb") as mock_emb, patch("worldline.llm.get_importance") as mock_imp:
        mock_emb.return_value = np.array([0.1, 0.2])
        mock_imp.return_value = 0.95
        
        note = MockNote(mock_ctx, "Hello world")
        
        # API should NOT have been called on init!
        mock_emb.assert_not_called()
        mock_imp.assert_not_called()
        
        # Accessing properties triggers evaluation
        assert note.importance == 0.95
        assert np.array_equal(note.emb, np.array([0.1, 0.2]))
        
        mock_emb.assert_called_once()
        mock_imp.assert_called_once()
        
        # Accessing again uses cache (call count remains 1)
        _ = note.importance
        _ = note.emb
        assert mock_imp.call_count == 1

def test_note_unpack_resets_lazy_cache(mock_ctx):
    with patch("worldline.llm.get_emb") as mock_emb, patch("worldline.llm.get_importance"):
        note = MockNote(mock_ctx, "State 1")
        
        # Trigger cache
        _ = note.emb
        assert mock_emb.call_count == 1
        
        # Unpack changes state and resets cache
        note.unpack("State 2")
        assert note.text == "State 2"
        
        # Cache should be cleared, next access triggers API again
        _ = note.emb
        assert mock_emb.call_count == 2

def test_note_equality_numpy_fix(mock_ctx):
    note1 = MockNote(mock_ctx, "A")
    note1.id = "ID-1"
    note1._emb = np.array([1, 2, 3])
    
    note2 = MockNote(mock_ctx, "B")
    note2.id = "ID-1"
    note2._emb = np.array([4, 5, 6])
    
    note3 = MockNote(mock_ctx, "C")
    note3.id = "ID-2"
    
    # This would crash with a ValueError without eq=False and the custom __eq__
    assert note1 == note2
    assert note1 != note3
    assert note1 != "not a note"

def test_note_time_travel_undo_redo(mock_ctx):
    # Setup initial state at Page 0
    note = MockNote(mock_ctx, "State 0")
    note.save()
    
    # Page 1
    mock_ctx.page_counter.step()
    note.text = "State 1"
    note.save()
    
    # Page 2
    mock_ctx.page_counter.step()
    note.text = "State 2"
    note.save()
    
    assert len(note.hist) == 3
    
    # Undo to Page 1
    mock_ctx.page_counter.page = 1
    note.sync()
    assert note.text == "State 1"
    assert len(note.hist) == 3 # Future is preserved!
    
    # Redo to Page 2
    mock_ctx.page_counter.page = 2
    note.sync()
    assert note.text == "State 2"
    
    # Undo to Page 1 and DIVERGE
    mock_ctx.page_counter.page = 1
    note.sync()
    note.text = "Alternate State 1"
    note.save()
    
    # Future should be erased, new path logged
    assert len(note.hist) == 2 
    assert note.hist[-1].state == "Alternate State 1"
