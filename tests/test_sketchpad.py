import pytest
from worldline.data import Context, PageCounter, Config
from worldline.uid import UIDGenerator
from worldline.sketchpad import Sketchpad

@pytest.fixture
def mock_ctx():
    return Context(
        uid_generator=UIDGenerator(0),
        page_counter=PageCounter(0),
        config=Config()
    )

def test_sketchpad_basic(mock_ctx):
    # Test initialization
    sp = Sketchpad(ctx=mock_ctx, name="Plan")
    assert sp.name == "Plan"
    assert sp.content == ""
    
    # Test _get_content
    content = sp.get_content(include_uid=False)
    assert "[Sketchpad] Plan:" in content
    
    # Test _tool_write directly
    res = sp._tool_write("This is a new plan.")
    assert "successfully" in res.lower()
    assert sp.content == "This is a new plan."
    assert sp.edited == True
    
    # Test tool property instantiation
    tool = sp.tool_write
    assert tool.name == "Plan - Write"
    
    # Test tool listing
    tools = sp.tools
    assert len(tools) == 1
    assert tools[0] == tool

def test_sketchpad_packing(mock_ctx):
    sp = Sketchpad(ctx=mock_ctx, name="Notes", content="Hello World")
    state = sp.pack()
    
    assert state["name"] == "Notes"
    assert state["content"] == "Hello World"
    
    # Test unpack
    sp2 = Sketchpad(ctx=mock_ctx)
    sp2.unpack(state)
    assert sp2.name == "Notes"
    assert sp2.content == "Hello World"
