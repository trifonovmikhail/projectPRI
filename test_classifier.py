import pytest
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from application import classify_text, load_model

@pytest.fixture
def tokenizer_and_model():
    tokenizer, model = load_model()
    return tokenizer, model

def test_load_model(tokenizer_and_model):
    tokenizer, model = tokenizer_and_model
    assert tokenizer is not None
    assert model is not None

def test_classify_text(tokenizer_and_model):
    tokenizer, model = tokenizer_and_model
    text = "Привет, как дела?"
    scores = classify_text(text, tokenizer, model)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 2