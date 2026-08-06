import pytest
import warnings
from worldline.data import Context, PageCounter, Config
from worldline.uid import UIDGenerator
from worldline.worldline import Worldline

@pytest.fixture
def mock_ctx():
    return Context(
        uid_generator=UIDGenerator(0),
        page_counter=PageCounter(0),
        config=Config(worldline_max_depth=3, worldline_recall_k=2)
    )

def test_worldline_initialization(mock_ctx):
    root = Worldline(ctx=mock_ctx, name="ROOT")
    assert root.name == "ROOT"
    assert root.open is True
    assert root.depth == 1
    assert root.can_dive is True
    assert root.latest is root
    # Entity edited defaults to True
    assert root.edited is True

def test_worldline_beat(mock_ctx):
    root = Worldline(ctx=mock_ctx, name="ROOT")
    root.edited = False # Reset for testing mutation
    
    root.beat("Action 1", "Did something.")
    
    assert len(root.children) == 1
    child = root[0]
    
    assert child.name == "Action 1"
    assert child.content == "Did something."
    assert child.open is False
    assert root.open is True # Root remains open
    assert root.depth == 1 # Depth is unaffected by beats
    
    # Verify edited flags!
    assert root.edited is True # Root appended a child, so it was edited

def test_worldline_dive_and_surface(mock_ctx):
    root = Worldline(ctx=mock_ctx, name="ROOT")
    root.edited = False
    
    root.dive("Arc 1")
    assert len(root.children) == 1
    arc = root[0]
    
    assert arc.name == "Arc 1"
    assert arc.open is True
    assert root.depth == 2
    assert root.latest is arc
    assert root.edited is True
    
    root.edited = False
    arc.edited = False
    
    root.surface("Arc 1 finished.")
    
    assert arc.open is False
    assert arc.content == "Arc 1 finished."
    assert root.depth == 1
    
    # Root didn't mutate its children array, the Arc mutated its own content!
    assert root.edited is False
    assert arc.edited is True

def test_worldline_routing(mock_ctx):
    root = Worldline(ctx=mock_ctx, name="ROOT")
    
    root.dive("Arc 1")
    arc1 = root[0]
    
    root.beat("Action 1", "Inside Arc 1.")
    assert len(arc1.children) == 1
    
    root.dive("Arc 2")
    arc2 = arc1[-1]
    
    root.beat("Action 2", "Inside Arc 2.")
    assert len(arc2.children) == 1
    
    # Root depth is 3 (Root -> Arc1 -> Arc2)
    assert root.depth == 3
    
    # Reset flags before testing deep beat append
    root.edited = False
    arc1.edited = False
    arc2.edited = False
    
    # Beat addition should route to Arc2
    root.beat("Action 3", "Inside Arc 2 again.")
    assert len(arc2.children) == 2
    
    # ONLY Arc2 should be flagged as edited! (O(n^2) recursion bug fix)
    assert root.edited is False
    assert arc1.edited is False
    assert arc2.edited is True

def test_worldline_errors(mock_ctx):
    root = Worldline(ctx=mock_ctx, name="ROOT")
    root.surface("Close ROOT")
    
    assert root.open is False
    
    with pytest.raises(RuntimeError):
        root.beat("Action", "Content")
        
    with pytest.raises(RuntimeError):
        root.dive("Arc")
        
    with pytest.raises(RuntimeError):
        root.surface("Already closed")

def test_worldline_max_depth_warning(mock_ctx):
    root = Worldline(ctx=mock_ctx, name="ROOT")
    
    root.dive("Depth 2")
    root.dive("Depth 3")
    
    assert root.can_dive is False
    
    # Diving past max depth raises a warning but allows it
    with pytest.warns(RuntimeWarning, match="Diving past maximum depth"):
        root.dive("Depth 4")
        
    assert root.depth == 4

def test_worldline_pack_unpack(mock_ctx):
    root = Worldline(ctx=mock_ctx, name="ROOT")
    root.dive("Arc 1")
    root.beat("Beat 1", "Action.")
    
    arc = root[0]
    beat = arc[0]
    
    packed_root = root.pack()
    assert packed_root["name"] == "ROOT"
    assert packed_root["child_UIDs"] == [arc.uid]
    
    packed_arc = arc.pack()
    assert packed_arc["child_UIDs"] == [beat.uid]
    
    # Test unpacking
    # Create a fresh root with no children
    new_root = Worldline(ctx=mock_ctx, name="New Root")
    new_root.unpack(packed_root)
    
    assert new_root.name == "ROOT"
    assert len(new_root.children) == 1
    assert new_root.children[0] == arc.uid

def test_worldline_get_content(mock_ctx):
    root = Worldline(ctx=mock_ctx, name="ROOT")
    root.dive("Arc 1")
    root.beat("Beat 1", "Action 1.")
    root.beat("Beat 2", "Action 2.")
    root.beat("Beat 3", "Action 3.")
    
    # By default, mock_ctx.config.worldline_recall_k is 2
    # So full=False should only show Beat 2 and Beat 3.
    content = root.get_content(include_uid=False)
    
    # The output should show Arc 1, Beat 2, Beat 3, and the >>> pointer
    assert "Arc 1" in content
    assert "Beat 2\nAction 2." in content
    assert "Beat 3\nAction 3." in content
    assert "Beat 1" not in content # Truncated!

def test_worldline_tools(mock_ctx):
    root = Worldline(ctx=mock_ctx, name="ROOT")
    
    # Test tool_beat
    res = root._tool_beat("Test Beat", "Some content.")
    assert "Success" in res
    assert len(root.children) == 1
    
    # Test length check
    long_content = "This is a sentence. " * (mock_ctx.config.worldline_max_entry_size + 5)
    res_err = root._tool_beat("Long Beat", long_content)
    assert "Error" in res_err
    assert "too long" in res_err
    
    # Test tool_dive
    res_dive = root._tool_dive("New Arc")
    assert "Success" in res_dive
    assert root.depth == 2
    
    # Test depth limit
    root._tool_dive("Depth 3")
    res_depth_err = root._tool_dive("Depth 4") # Max depth is 3 in mock_ctx
    assert "Error" in res_depth_err
    assert "max depth" in res_depth_err
    
    # Test tool_surface
    res_surf = root._tool_surface("Finished depth 3")
    assert "Success" in res_surf
    assert root.depth == 2
    
    # Test tool properties
    tools = root.tools
    assert len(tools) == 3
    assert tools[0].name == "ROOT - Beat"
    assert tools[1].name == "ROOT - Dive"
    assert tools[2].name == "ROOT - Surface"
