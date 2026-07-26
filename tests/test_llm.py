import pytest
from unittest.mock import patch, MagicMock
import worldline.llm as llm

@patch("os.environ", {"GEMINI_API_KEY": "fake_key"})
@patch("dotenv.load_dotenv")
@patch("dspy.LM")
@patch("dspy.Embedder")
@patch("dspy.Predict")
def test_init_sets_globals(mock_predict, mock_embedder, mock_lm, mock_dotenv):
    # Reset globals in case other tests ran
    llm.LM_LIGHT = None
    llm.LM_HEAVY = None
    llm.EMB = None
    llm.importance = None
    
    mock_lm.side_effect = ["light_model", "heavy_model"]
    mock_embedder.return_value = "embedder_model"
    mock_predict.return_value = "predict_module"
    
    llm.init()
    
    assert llm.GEMINI_API_KEY == "fake_key"
    assert llm.LM_LIGHT == "light_model"
    assert llm.LM_HEAVY == "heavy_model"
    assert llm.EMB == "embedder_model"
    assert llm.importance == "predict_module"
    
    # Test idempotency
    llm.init()
    assert mock_lm.call_count == 2 # 2 calls from the first init (light, heavy), shouldn't increment

@patch("worldline.llm.init")
def test_get_importance(mock_init):
    # Set up mock importance module
    mock_importance_module = MagicMock()
    
    # Mock the return value chain: importance_module(text=..., lm=...).importance
    mock_result = MagicMock()
    mock_result.importance = 0.95
    mock_importance_module.return_value = mock_result
    
    llm.importance = mock_importance_module
    llm.LM_LIGHT = "mock_lm_light"
    
    # Clear lru_cache for testing
    llm.get_importance.cache_clear()
    
    result = llm.get_importance("Test text")
    
    assert result == 0.95
    mock_init.assert_called_once()
    mock_importance_module.assert_called_once_with(text="Test text", lm="mock_lm_light")

@patch("worldline.llm.init")
def test_get_emb(mock_init):
    mock_emb_module = MagicMock()
    mock_emb_module.return_value = [0.1, 0.2, 0.3]
    
    llm.EMB = mock_emb_module
    
    # Clear lru_cache for testing
    llm.get_emb.cache_clear()
    
    result = llm.get_emb("Test text")
    
    assert result == [0.1, 0.2, 0.3]
    mock_init.assert_called_once()
    mock_emb_module.assert_called_once_with("Test text")
