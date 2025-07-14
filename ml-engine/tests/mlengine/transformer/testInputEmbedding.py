import torch
import sys
import os
import numpy as np
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mlengine.adapters.transformer.SRC.InputEmbedding import InputEmbeddings

@pytest.mark.parametrize("action_count, embedding_size", [(10, 4), (20, 8)])
def test_input_embedding(action_count, embedding_size):
    embedding_model = InputEmbeddings(action_count, embedding_size)
    dummy_input = torch.LongTensor([[1, 3, 5], [2, 4, 6]])
    output = embedding_model(dummy_input)
    assert output.shape == (2, 3, embedding_size)
    assert output.dtype == torch.float32