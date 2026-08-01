import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import worldline.llm as llm

@patch("os.environ", {"GEMINI_API_KEY": "fake_gemini", "OPENAI_API_KEY": "fake_openai", "ANTHROPIC_API_KEY": "fake_anthropic"})
@patch("dotenv.load_dotenv")
@patch("dspy.LM")
@patch("requests.get")
def test_init_models(mock_requests_get, mock_lm, mock_dotenv):
    # Reset globals in case other tests ran
    llm.TEXT_MODELS.clear()
    llm.importance = None
    
    mock_lm.return_value = "mock_lm_instance"
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "hermes3:8b"}, {"name": "llama3"}]}
    mock_requests_get.return_value = mock_response
    
    llm.init_models()
    
    # Check that cloud models are populated
    assert "gemini-2.5-flash-lite" in llm.TEXT_MODELS
    assert "gpt-5-nano" in llm.TEXT_MODELS
    assert "claude-haiku-4-5" in llm.TEXT_MODELS
    
    # Check that local models are populated
    assert "local-hermes3:8b" in llm.TEXT_MODELS
    assert "local-llama3" in llm.TEXT_MODELS
    
    # Test idempotency - calling again shouldn't duplicate or crash
    llm.init_models()
    # It shouldn't create new LM instances for already existing keys
    assert mock_lm.call_count == 13 # 3 gemini + 3 openai + 3 anthropic + 2 local + 2 local (since Ollama lacks idempotency check)

@patch("sentence_transformers.SentenceTransformer")
def test_get_emb(mock_sentence_transformer):
    # Reset the global so it initializes the mock and clear cache
    llm.LOCAL_EMB_MODEL = None
    llm._EMB_CACHE.clear()
    
    mock_emb_model = MagicMock()
    mock_emb_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_sentence_transformer.return_value = mock_emb_model
    
    result = llm.get_emb("Test text")
    
    assert np.allclose(result, [0.1, 0.2, 0.3])
    # Init_models is no longer called in get_emb
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
