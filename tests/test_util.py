import dataclasses
from unittest.mock import MagicMock, patch
import pytest

import worldline as wl
from worldline.util import get_page, PageCounter, Note

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

def test_get_page():
    assert get_page() == -1
    assert get_page(None) == -1
    assert get_page(5) == 5
    pc = PageCounter(7)
    assert get_page(pc) == 7

def test_note_init_default_page():
    note = Note("test_note", "test_content")
    assert note.name == "test_note"
    assert note.content == "test_content"
    assert note.page == -1

def test_note_init_int_page():
    note = Note("test_note", "test_content", page=5)
    assert note.page == 5

def test_note_init_pagecounter_page():
    pc = PageCounter(7)
    note = Note("test_note", "test_content", page=pc)
    assert note.page == 7

def test_note_str():
    note = Note("My Note", "Some content")
    assert str(note) == "My Note: Some content"

def test_note_is_frozen():
    note = Note("test", "content")
    with pytest.raises(dataclasses.FrozenInstanceError):
        note.name = "new_name"

def test_note_copy_with():
    original = Note("Idea", "Apples are red", page=10)
    modified = original.copy_with(content="Apples are green")
    
    assert original.content == "Apples are red"
    assert modified.name == "Idea"
    assert modified.content == "Apples are green"
    assert modified.page == 10
    
    modified_page = original.copy_with(page=20)
    assert modified_page.page == 20

@patch("worldline.util.get_emb")
def test_note_emb_property(mock_get_emb):
    mock_get_emb.return_value = [0.1, 0.2, 0.3]
    note = Note("test", "content")
    assert note.emb == [0.1, 0.2, 0.3]
    mock_get_emb.assert_called_with("test: content")

@patch("worldline.util.get_importance")
def test_note_importance_property(mock_get_importance):
    mock_get_importance.return_value = 0.85
    note = Note("test", "content")
    assert note.importance == 0.85
    mock_get_importance.assert_called_with("content")
