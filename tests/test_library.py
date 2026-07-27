import pytest
from worldline.data import Context, PageCounter, Config
from worldline.uid import UIDGenerator
from worldline.library import Record

@pytest.fixture
def mock_ctx():
    return Context(
        uid_generator=UIDGenerator(0),
        page_counter=PageCounter(0),
        config=Config()
    )

def test_record_initialization(mock_ctx):
    record = Record(
        ctx=mock_ctx,
        title="Safe Code",
        content="The code is 1234",
        source="Hunch"
    )
    
    assert record.title == "Safe Code"
    assert record.content == "The code is 1234"
    assert record.source == "Hunch"
    assert record.related == []
    
    # Verify inheritance setup (registers automatically with Context)
    assert record.uid is not None
    assert mock_ctx.registry[record.uid] is record

def test_record_get_content_formatting(mock_ctx):
    # Create a related record to test dynamic title resolution
    related_record = Record(
        ctx=mock_ctx,
        title="The Safe",
        content="It's in the basement."
    )
    
    main_record = Record(
        ctx=mock_ctx,
        title="Safe Code",
        content="The code is 1234",
        source="Found a note",
        related=[related_record.uid]
    )
    
    content_str = main_record.get_content(include_uid=True)
    
    # Should include UID prefix natively from Note inheritance
    assert content_str.startswith(f"[UID {main_record.uid}]")
    
    # Should format fields correctly
    assert "Safe Code:" in content_str
    assert "The code is 1234" in content_str
    assert "Source: Found a note" in content_str
    
    # Most importantly, it should have dynamically fetched the title of the related record!
    expected_related_str = f"Related: [UID {related_record.uid}] The Safe"
    assert expected_related_str in content_str

def test_record_pack_unpack(mock_ctx):
    record = Record(
        ctx=mock_ctx,
        title="Original Title",
        content="Original Content",
        source="Original Source",
        related=["UID-X", "UID-Y"]
    )
    
    state = record.pack()
    
    assert state["title"] == "Original Title"
    assert state["content"] == "Original Content"
    assert state["source"] == "Original Source"
    assert state["related_UIDs"] == ["UID-X", "UID-Y"]
    
    # Create a blank record and unpack over it
    blank_record = Record(ctx=mock_ctx)
    blank_record.unpack(state)
    
    assert blank_record.title == "Original Title"
    assert blank_record.content == "Original Content"
    assert blank_record.source == "Original Source"
    assert blank_record.related == ["UID-X", "UID-Y"]
