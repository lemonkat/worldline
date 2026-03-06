import os
import sys
from unittest.mock import MagicMock, patch
import pytest

import worldline as wl

def test_page_counter_init_default():
    pc = wl.PageCounter()
    assert pc.page == 0

def test_page_counter_init_custom():
    pc = wl.PageCounter(10)
    assert pc.page == 10

def test_page_counter_step():
    pc = wl.PageCounter(0)
    assert pc.step() == 1
    assert pc.page == 1
    assert pc.step() == 2

def test_page_counter_str():
    pc = wl.PageCounter(5)
    assert str(pc) == "Page 5"

def test_page_counter_repr():
    pc = wl.PageCounter(5)
    assert repr(pc) == "PageCounter(5)"

def test_note_init_default_page():
    note = wl.Note("test_note", "test_content")
    assert note.name == "test_note"
    assert note.content == "test_content"
    assert note.pages == [0]

def test_note_init_int_page():
    note = wl.Note("test_note", "test_content", page=5)
    assert note.pages == [5]

def test_note_init_list_page():
    note = wl.Note("test_note", "test_content", page=[1, 2, 3])
    assert note.pages == [1, 2, 3]

def test_note_init_pagecounter_page():
    pc = wl.PageCounter(7)
    note = wl.Note("test_note", "test_content", page=pc)
    assert note.pages == [7]

def test_note_str():
    note = wl.Note("My Note", "Some content")
    assert str(note) == "My Note: Some content"

def test_note_emb_property():
    note = wl.Note("test", "content")
    with patch("worldline.util.get_emb", return_value=[0.1, 0.2, 0.3]) as mock_get_emb:
        assert note.emb == [0.1, 0.2, 0.3]
        mock_get_emb.assert_called_with("content")

def test_note_importance_property():
    note = wl.Note("test", "content")
    mock_result = MagicMock()
    with patch("worldline.util.get_importance", return_value=mock_result) as mock_get_importance:
        assert note.importance == mock_result
        mock_get_importance.assert_called_with("content")

def test_get_importance():
    mock_result = MagicMock()
    mock_result.value = 0.85
    with patch("worldline.util.init"), patch("worldline.util.importance", return_value=mock_result) as mock_pred:
        assert wl.get_importance("test text") == 0.85
        mock_pred.assert_called_with("test text")

def test_recall_settings_defaults():
    settings = wl.RecallSettings()
    assert settings.weight_sim == 0.5
    assert settings.weight_recency == 0.25
    assert settings.weight_importance == 0.25
    assert settings.page_decay == 0.99

def test_recall_settings_custom():
    settings = wl.RecallSettings(
        weight_sim=0.1,
        weight_recency=0.2,
        weight_importance=0.3,
        page_decay=0.4
    )
    assert settings.weight_sim == 0.1
    assert settings.weight_recency == 0.2
    assert settings.weight_importance == 0.3
    assert settings.page_decay == 0.4
