import pytest
from unittest.mock import patch
import operator
import numpy as np

from worldline.data import PageCounter, Note, Context, Config
from worldline.uid import UIDGenerator

# --- PageCounter Tests ---

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


# --- Context and Note Tests ---

class MockNote(Note):
    text: str = ""
        
    def _get_content(self):
        return self.text
        
    def _pack(self):
        return {"text": self.text}
        
    def _unpack(self, state):
        self.text = state["text"]

@pytest.fixture
def mock_ctx():
    uid_gen = UIDGenerator(0)
    pc = PageCounter(0)
    return Context(uid_generator=uid_gen, page_counter=pc, config=Config())

def test_entity_setup_and_registry(mock_ctx):
    # Note should automatically get a UID and register itself
    note = MockNote(ctx=mock_ctx, text="Hello")
    assert note.uid is not None
    assert isinstance(note.uid, str)
    assert mock_ctx.registry[note.uid] is note

def test_context_record_and_rewind(mock_ctx):
    # Setup initial state at Page 0
    note1 = MockNote(ctx=mock_ctx, text="State 0")
    note1.edited = True
    mock_ctx.record()
    
    assert len(mock_ctx.history) == 1
    assert mock_ctx.history[0].page == 0
    
    # Page 1
    mock_ctx.page_counter.step()
    note1.text = "State 1"
    note1.edited = True
    note2 = MockNote(ctx=mock_ctx, text="New Object")
    note2.edited = True
    mock_ctx.record()
    
    assert len(mock_ctx.history) == 3 # Update(0, uid0), Update(1, uid0), Update(1, uid1)
    
    # Page 2
    mock_ctx.page_counter.step()
    note1.text = "State 2"
    note1.edited = True
    mock_ctx.record()
    
    assert len(mock_ctx.history) == 4
    
    # Undo to Page 1
    mock_ctx.page_counter.page = 1
    mock_ctx.rewind()
    assert note1.text == "State 1"
    assert note2.text == "New Object"
    assert len(mock_ctx.history) == 4 # Future is preserved!
    
    # Undo to Page 0
    mock_ctx.page_counter.page = 0
    mock_ctx.rewind()
    assert note1.text == "State 0"
    # note2's text remains "New Object" because it was unborn at Page 0 and ignored by rewind
    
    # Undo to Page 1 and DIVERGE (Erase the future)
    mock_ctx.page_counter.page = 1
    mock_ctx.rewind()
    assert note1.text == "State 1"
    
    note1.text = "Alternate State 1"
    note1.edited = True
    mock_ctx.record()
    
    # The history should have popped the old Page 2 update, and added the new Page 1 update
    # old len was 4. Pop 1. Append 1. New len 4.
    assert len(mock_ctx.history) == 4 
    assert mock_ctx.history[-1].state["text"] == "Alternate State 1"
    assert mock_ctx.history[-1].page == 1



def test_note_get_content(mock_ctx):
    note = MockNote(ctx=mock_ctx, text="Test content")
    assert note.get_content(include_uid=False) == "Test content"
    assert note.get_content(include_uid=True) == f"[UID {note.uid}] Test content"

def test_context_concurrency(mock_ctx):
    import threading

    def worker():
        for _ in range(100):
            with mock_ctx.lock:
                mock_ctx.page_counter.step()
                note = MockNote(ctx=mock_ctx, text="Concurrency test")
                note.edited = True
                mock_ctx.record()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 10 threads doing 100 steps each = 1000 history entries
    assert len(mock_ctx.history) == 1000
    assert mock_ctx.page_counter.page == 1000

def test_note_from_packed(mock_ctx):
    # Setup state to restore from
    uid = mock_ctx.uid_generator.next()
    state = {"text": "Restored State"}
    
    # Restore from packed state
    note = MockNote.from_packed(ctx=mock_ctx, uid=uid, state=state)
    
    assert note.uid == uid
    assert note.text == "Restored State"
    assert note.edited is False
    assert mock_ctx.registry[uid] is note
