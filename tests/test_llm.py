import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import worldline.llm as llm

@patch("os.environ", {"GEMINI_API_KEY": "fake_key"})
@patch("dotenv.load_dotenv")
@patch("dspy.LM")
@patch("dspy.Predict")
def test_init_sets_globals(mock_predict, mock_lm, mock_dotenv):
    # Reset globals in case other tests ran
    llm.LM_LIGHT = None
    llm.LM_HEAVY = None
    llm.importance = None
    
    mock_lm.side_effect = ["light_model", "heavy_model"]
    mock_predict.return_value = "predict_module"
    
    llm.init()
    
    assert llm.GEMINI_API_KEY == "fake_key"
    assert llm.LM_LIGHT == "light_model"
    assert llm.LM_HEAVY == "heavy_model"
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
    
    result = llm.get_importance("Test text")
    
    assert result == 0.95
    mock_init.assert_called_once()
    mock_importance_module.assert_called_once_with(text="Test text", lm="mock_lm_light")

@patch("worldline.llm.init")
@patch("dspy.context")
@patch("dspy.Parallel")
def test_get_importance_batch(mock_parallel, mock_context, mock_init):
    import worldline.llm as llm
    mock_parallel_instance = MagicMock()
    mock_result1 = MagicMock()
    mock_result1.importance = 0.8
    mock_result2 = MagicMock()
    mock_result2.importance = 0.9
    mock_parallel_instance.return_value = [mock_result1, mock_result2]
    mock_parallel.return_value = mock_parallel_instance
    
    result = llm.get_importance(["text1", "text2"])
    
    assert result == [0.8, 0.9]
    mock_parallel.assert_called_once_with(num_threads=15, disable_progress_bar=True)
    mock_parallel_instance.assert_called_once()

@patch("worldline.llm.init")
@patch("sentence_transformers.SentenceTransformer")
def test_get_emb(mock_sentence_transformer, mock_init):
    # Reset the global so it initializes the mock and clear cache
    llm.LOCAL_EMB_MODEL = None
    llm._EMB_CACHE.clear()
    
    mock_emb_model = MagicMock()
    mock_emb_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_sentence_transformer.return_value = mock_emb_model
    
    result = llm.get_emb("Test text")
    
    assert np.allclose(result, [0.1, 0.2, 0.3])
    mock_init.assert_called_once()
    mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
    mock_emb_model.encode.assert_called_once_with(["Test text"], convert_to_numpy=True)

@patch("nltk.download")
def test_count_sentences(mock_download):
    # Reset the initialization flag and cache for testing
    llm._NLTK_INITIALIZED = False
    llm.count_sentences.cache_clear()
    
    # Test simple sentences
    text1 = "This is a sentence. This is another."
    assert llm.count_sentences(text1) == 2
    mock_download.assert_called_once_with('punkt', quiet=True)
    
    # Test that download is not called again because the flag is set
    mock_download.reset_mock()
    text2 = "Just one."
    assert llm.count_sentences(text2) == 1
    mock_download.assert_not_called()
    
    # Test edge cases (Mr. Smith, ellipses) that would break simple string splits
    text3 = "Mr. Smith went to the store. He bought apples... lots of them! Was it a good day? Yes."
    assert llm.count_sentences(text3) == 4
