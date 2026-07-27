import pytest
import numpy as np
from unittest.mock import patch

from worldline.data import Context, PageCounter, Config
from worldline.uid import UIDGenerator
from worldline.library import Record, Library, MissingRecordError, UnloadedRecordError

@pytest.fixture
def mock_ctx():
    return Context(
        uid_generator=UIDGenerator(0),
        page_counter=PageCounter(0),
        config=Config()
    )

def test_library_create_update_delete(mock_ctx):
    library = Library(ctx=mock_ctx, title="Test Lib")
    
    # Test Create
    record = library.create(
        title="Apples",
        content="Apples are red.",
        source="Farmer",
        related=[]
    )
    assert record.uid in library.records
    assert library.records[record.uid].title == "Apples"
    assert library.edited is True
    
    # Test Update (Unloaded should fail)
    with pytest.raises(UnloadedRecordError):
        library.update(uid=record.uid, content=" Wait, some are green.")
        
    # Test Update (Loaded should succeed)
    library.recall(record.uid)
    library.edited = False # Reset for testing
    record.edited = False
    
    library.update(uid=record.uid, content=" Wait, some are green.", append=True)
    assert library.records[record.uid].content == "Apples are red. Wait, some are green."
    assert library.edited is True
    assert record.edited is True
    
    # Test Overwrite (append=False)
    library.update(uid=record.uid, content="Only green apples.", append=False)
    assert library.records[record.uid].content == "Only green apples."
    
    # Test Delete (Unloaded should fail)
    library.refresh() # Clear loaded
    with pytest.raises(UnloadedRecordError):
        library.delete(uid=record.uid)
        
    # Test Delete (Loaded should succeed)
    library.recall(record.uid)
    library.delete(uid=record.uid)
    assert record.uid not in library.records

def test_library_refresh_and_recall(mock_ctx):
    library = Library(ctx=mock_ctx)
    r1 = library.create(title="R1", content="...")
    r2 = library.create(title="R2", content="...")
    
    # Recall sets the score to 1e6
    library.recall(r1.uid)
    assert r1.uid in library.loaded
    assert library.loaded[r1.uid] == 1e6
    
    # Refresh clears working memory
    library.refresh()
    assert len(library.loaded) == 0
    assert r1.uid not in library.loaded
    
    # Recall non-existent raises error
    with pytest.raises(MissingRecordError):
        library.recall("FAKE-UID")

def test_library_search(mock_ctx):
    with patch("worldline.library.get_emb") as mock_emb, patch("worldline.llm.get_importance") as mock_imp:
        # Mock embeddings to be predictable
        mock_emb.side_effect = lambda text: np.array([1.0, 0.0]) if "magic" in text else np.array([0.0, 1.0])
        mock_imp.return_value = 0.5
        
        library = Library(ctx=mock_ctx)
        mock_ctx.config.library_search_k = 1 

        r_magic = library.create(title="Sword", content="magic sword")
        r_normal = library.create(title="Rock", content="normal rock")
        
        # Explicitly set embeddings to avoid Note property lazy-evaluation calling original get_emb
        r_magic._emb = np.array([1.0, 0.0])
        r_normal._emb = np.array([0.0, 1.0])
        r_magic._importance = 0.5
        r_normal._importance = 0.5
        
        # Search for magic (should score high on r_magic)
        results = library.search("I want magic", mark_loaded=True)
        assert len(results) > 0
        assert results[0].uid == r_magic.uid
        
        # Verify mark_loaded worked
        assert r_magic.uid in library.loaded
        assert r_normal.uid not in library.loaded
        
        # Search again for magic (r_magic is already loaded, should return r_normal)
        results2 = library.search("I want magic", mark_loaded=True)
        assert len(results2) > 0
        assert results2[0].uid == r_normal.uid

def test_library_time_travel(mock_ctx):
    library = Library(ctx=mock_ctx, title="Time Lib")
    
    # PAGE 0: Initial State
    record = library.create(title="Origin", content="Start")
    library.recall(record.uid)
    mock_ctx.record()
    
    # PAGE 1: First Edit
    mock_ctx.page_counter.step()
    library.update(uid=record.uid, title="Future", content=" Next", append=True)
    mock_ctx.record()
    
    assert library.records[record.uid].title == "Future"
    assert library.records[record.uid].content == "Start Next"
    # Registry has Library, Record. Page 0 writes both (since edited=True upon creation/recall). Page 1 writes both.
    assert len(mock_ctx.history) == 4 
    
    # PAGE 2: Delete
    mock_ctx.page_counter.step()
    library.delete(uid=record.uid)
    mock_ctx.record()
    
    assert record.uid not in library.records
    
    # TIME TRAVEL: REWIND TO PAGE 1
    mock_ctx.page_counter.page = 1
    mock_ctx.rewind()
    
    assert record.uid in library.records # Un-deleted!
    assert library.records[record.uid].title == "Future"
    assert library.records[record.uid].content == "Start Next"
    
    # TIME TRAVEL: REWIND TO PAGE 0
    mock_ctx.page_counter.page = 0
    mock_ctx.rewind()
    
    assert library.records[record.uid].title == "Origin"
    assert library.records[record.uid].content == "Start"

def test_record_lazy_evaluation(mock_ctx):
    with patch("worldline.llm.get_emb") as mock_emb, patch("worldline.llm.get_importance") as mock_imp:
        mock_emb.return_value = np.array([0.1, 0.2])
        mock_imp.return_value = 0.95
        
        record = Record(ctx=mock_ctx, title="Title", content="Hello world")
        
        # API should NOT have been called on init!
        mock_emb.assert_not_called()
        mock_imp.assert_not_called()
        
        # Accessing properties triggers evaluation
        assert record.importance == 0.95
        assert np.array_equal(record.emb, np.array([0.1, 0.2]))
        
        mock_emb.assert_called_once()
        mock_imp.assert_called_once()
        
        # Accessing again uses cache (call count remains 1)
        _ = record.importance
        _ = record.emb
        assert mock_imp.call_count == 1

def test_record_unpack_resets_lazy_cache(mock_ctx):
    with patch("worldline.llm.get_emb") as mock_emb, patch("worldline.llm.get_importance"):
        record = Record(ctx=mock_ctx, title="State 1", content="")
        
        # Trigger cache
        _ = record.emb
        assert mock_emb.call_count == 1
        
        # Unpack changes state and resets cache
        record.unpack({"title": "State 2", "content": "", "source": None, "related_UIDs": []})
        assert record.title == "State 2"
        
        # Cache should be cleared, next access triggers API again
        _ = record.emb
        assert mock_emb.call_count == 2

def test_record_equality_numpy_fix(mock_ctx):
    record1 = Record(ctx=mock_ctx, title="A")
    record1.uid = "ID-1"
    record1._emb = np.array([1, 2, 3])
    
    record2 = Record(ctx=mock_ctx, title="B")
    record2.uid = "ID-1"
    record2._emb = np.array([4, 5, 6])
    
    record3 = Record(ctx=mock_ctx, title="C")
    record3.uid = "ID-2"
    
    # This would crash with a ValueError without eq=False and the custom __eq__
    assert record1 == record2
    assert record1 != record3
    assert record1 != "not a record"
