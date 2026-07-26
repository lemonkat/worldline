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

@patch("worldline.llm.get_emb")
@patch("worldline.llm.get_importance")
def test_note_init_fetches_defaults(mock_get_imp, mock_get_emb):
    mock_get_emb.return_value = [0.1, 0.2]
    mock_get_imp.return_value = 0.85
    
    note = Note(id="TEST-1234", content="Hello world", page=1)
    
    assert note.id == "TEST-1234"
    assert note.content == "Hello world"
    assert note.page == 1
    assert note.emb == [0.1, 0.2]
    assert note.importance == 0.85
    
    mock_get_emb.assert_called_once_with("Hello world")
    mock_get_imp.assert_called_once_with("Hello world")

@patch("worldline.llm.get_emb")
@patch("worldline.llm.get_importance")
def test_note_init_provided_values_skips_api(mock_get_imp, mock_get_emb):
    note = Note(
        id="TEST-1234", 
        content="Hello world", 
        page=1,
        emb=[0.9, 0.9],
        importance=0.1
    )
    
    assert note.emb == [0.9, 0.9]
    assert note.importance == 0.1
    
    mock_get_emb.assert_not_called()
    mock_get_imp.assert_not_called()
