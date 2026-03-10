import pytest
from unittest.mock import patch
import worldline as wl

def test_library_init():
    lib = wl.Library("test_lib")
    assert lib.name == "test_lib"
    assert lib.notes == []
    assert lib.weight_relevance == 2.0
    assert lib.weight_importance == 1.0
    assert lib.summary_size == 10

def test_library_init_with_notes():
    note1 = wl.Note("1", "one")
    note2 = wl.Note("2", "two")
    lib = wl.Library("test_lib", notes=[note1, note2], summary_size=5)
    assert len(lib.notes) == 2
    assert lib.summary_size == 5

def test_library_add():
    lib = wl.Library("test_lib")
    note = wl.Note("1", "one")
    lib.add(note)
    assert len(lib.notes) == 1
    assert lib.notes[0] == note

def test_library_str():
    lib = wl.Library("My Library")
    assert str(lib) == "Library: My Library"

@patch("worldline.util.get_emb")
@patch("worldline.util.get_importance")
def test_library_score(mock_get_imp, mock_get_emb):
    def mock_emb(text):
        if "query" in text.lower():
            return [1.0, 0.0]
        else:
            return [0.0, 1.0]
            
    mock_get_emb.side_effect = mock_emb
    mock_get_imp.return_value = 0.5
    
    lib = wl.Library("test", weight_relevance=2.0, weight_importance=1.0)
    query = wl.Note("query", "Im a query")
    item = wl.Note("item", "Im an item")
    
    assert lib.score(query, item) == 0.5
    assert lib.score(None, item) == 0.5
    assert lib.score("query str", item) == 0.5

@patch("worldline.util.get_emb")
@patch("worldline.util.get_importance")
def test_library_search(mock_get_imp, mock_get_emb):
    note1 = wl.Note("1", "bad match")
    note2 = wl.Note("2", "good match")
    note3 = wl.Note("3", "okay match")
    
    lib = wl.Library("test_lib", notes=[note1, note2, note3])
    
    with patch.object(lib, 'score') as mock_score:
        def mock_score_func(q, item):
            if item == note1: return 0.1
            if item == note2: return 0.9
            if item == note3: return 0.5
            return 0.0
        mock_score.side_effect = mock_score_func
        
        results = lib.search("query", k=2)
        assert len(results) == 2
        assert results[0] == note2
        assert results[1] == note3
        
        results_none = lib.search(None, k=2)
        assert results_none == [note2, note3]

@patch("worldline.util.get_emb")
@patch("worldline.util.get_importance")
def test_library_summary(mock_get_imp, mock_get_emb):
    note1 = wl.Note("1", "content 1", page=0)
    note2 = wl.Note("2", "content 2", page=1)
    lib = wl.Library("TestLib", notes=[note1, note2], summary_size=2)
    
    with patch.object(lib, 'search') as mock_search:
        mock_search.return_value = [note2, note1]
        summary_note = lib.summary
        
        assert summary_note.name == "Pertinent information relating to TestLib:"
        assert summary_note.page == -1
        assert "2: content 2" in summary_note.content
        assert "1: content 1" in summary_note.content
        mock_search.assert_called_with(None, 2)

def test_worldline_init():
    wl_lib = wl.Worldline("test_wl")
    assert wl_lib.weight_recency == 1.0
    assert wl_lib.t_decay == 0.95
    assert wl_lib.t_global_base == 0.5

@patch("worldline.util.get_emb")
@patch("worldline.util.get_importance")
def test_worldline_score(mock_get_imp, mock_get_emb):
    mock_get_emb.return_value = [0.0, 0.0]
    mock_get_imp.return_value = 0.5
    
    wl_lib = wl.Worldline(
        "test_wl", 
        weight_relevance=0.0, 
        weight_importance=1.0, 
        weight_recency=2.0, 
        t_decay=0.9, 
        t_global_base=0.4
    )
    
    item = wl.Note("item1", "test", page=5)
    
    assert wl_lib.score("query", item) == 0.5
    assert wl_lib.score(None, item) == 0.5
    
    query1 = wl.Note("q1", "q1", page=10)
    assert wl_lib.score(query1, item) == pytest.approx(0.5 + (0.9 ** 5) * 2.0)
    
    query2 = wl.Note("q2", "q2", page=-1)
    assert wl_lib.score(query2, item) == pytest.approx(0.5 + 0.4 * 2.0)
    
    item_timeless = wl.Note("item2", "test", page=-1)
    assert wl_lib.score(query1, item_timeless) == pytest.approx(0.5 + 0.4 * 2.0)
