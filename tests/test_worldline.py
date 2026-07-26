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
    root = Worldline(ctx=mock_ctx, title="ROOT")
    assert root.title == "ROOT"
    assert root.open is True
    assert root.depth == 1
    assert root.can_dive is True
    assert root.latest is root
    # Entity edited defaults to True
    assert root.edited is True

def test_worldline_step(mock_ctx):
    root = Worldline(ctx=mock_ctx, title="ROOT")
    root.edited = False # Reset for testing mutation
    
    root.step("Action 1", "Did something.")
    
    assert len(root.children) == 1
    child = root.children[0]
    
    assert child.title == "Action 1"
    assert child.content == "Did something."
    assert child.open is False
    assert root.open is True # Root remains open
    assert root.depth == 1 # Depth is unaffected by beats
    
    # Verify edited flags!
    assert root.edited is True # Root appended a child, so it was edited

def test_worldline_dive_and_surface(mock_ctx):
    root = Worldline(ctx=mock_ctx, title="ROOT")
    root.edited = False
    
    root.dive("Arc 1")
    assert len(root.children) == 1
    arc = root.children[0]
    
    assert arc.title == "Arc 1"
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
    root = Worldline(ctx=mock_ctx, title="ROOT")
    
    root.dive("Arc 1")
    arc1 = root.children[0]
    
    root.step("Action 1", "Inside Arc 1.")
    assert len(arc1.children) == 1
    
    root.dive("Arc 2")
    arc2 = arc1.children[-1]
    
    root.step("Action 2", "Inside Arc 2.")
    assert len(arc2.children) == 1
    
    # Root depth is 3 (Root -> Arc1 -> Arc2)
    assert root.depth == 3
    
    # Reset flags before testing deep step
    root.edited = False
    arc1.edited = False
    arc2.edited = False
    
    # Stepping should route to Arc2
    root.step("Action 3", "Inside Arc 2 again.")
    assert len(arc2.children) == 2
    
    # ONLY Arc2 should be flagged as edited! (O(n^2) recursion bug fix)
    assert root.edited is False
    assert arc1.edited is False
    assert arc2.edited is True

def test_worldline_errors(mock_ctx):
    root = Worldline(ctx=mock_ctx, title="ROOT")
    root.surface("Close ROOT")
    
    assert root.open is False
    
    with pytest.raises(RuntimeError):
        root.step("Action", "Content")
        
    with pytest.raises(RuntimeError):
        root.dive("Arc")
        
    with pytest.raises(RuntimeError):
        root.surface("Already closed")

def test_worldline_max_depth_warning(mock_ctx):
    root = Worldline(ctx=mock_ctx, title="ROOT")
    
    root.dive("Depth 2")
    root.dive("Depth 3")
    
    assert root.can_dive is False
    
    # Diving past max depth raises a warning but allows it
    with pytest.warns(RuntimeWarning, match="Diving past maximum depth"):
        root.dive("Depth 4")
        
    assert root.depth == 4

def test_worldline_pack_unpack(mock_ctx):
    root = Worldline(ctx=mock_ctx, title="ROOT")
    root.dive("Arc 1")
    root.step("Beat 1", "Action.")
    
    arc = root.children[0]
    beat = arc.children[0]
    
    packed_root = root.pack()
    assert packed_root["title"] == "ROOT"
    assert packed_root["child_UIDs"] == [arc.uid]
    
    packed_arc = arc.pack()
    assert packed_arc["child_UIDs"] == [beat.uid]
    
    # Test unpacking
    # Create a fresh root with no children
    new_root = Worldline(ctx=mock_ctx, title="New Root")
    new_root.unpack(packed_root)
    
    assert new_root.title == "ROOT"
    assert len(new_root.children) == 1
    assert new_root.children[0] is arc

def test_worldline_get_content(mock_ctx):
    root = Worldline(ctx=mock_ctx, title="ROOT")
    root.dive("Arc 1")
    root.step("Beat 1", "Action 1.")
    root.step("Beat 2", "Action 2.")
    root.step("Beat 3", "Action 3.")
    
    # By default, mock_ctx.config.worldline_recall_k is 2
    # So full=False should only show Beat 2 and Beat 3.
    content = root.get_content(include_uid=False)
    
    # The output should show Arc 1, Beat 2, Beat 3, and the >>> pointer
    assert "Arc 1" in content
    assert "Beat 2: Action 2." in content
    assert "Beat 3: Action 3." in content
    assert "Beat 1" not in content # Truncated!
    assert ">>>" in content
