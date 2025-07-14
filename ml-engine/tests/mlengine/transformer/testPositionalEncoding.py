import torch
import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mlengine.adapters.transformer.SRC.PositionalEncoding import PositionalEncoding

@pytest.mark.parametrize("embedding_size, seq_len, batch_size", [(16, 10, 4)])
def test_positional_encoding(embedding_size, seq_len, batch_size):
    pos_enc = PositionalEncoding(embedding_size, dropout=0.1, sequence_max_length=100)
    dummy_input = torch.zeros(seq_len, batch_size, embedding_size)
    output = pos_enc(dummy_input)
    assert output.shape == (seq_len, batch_size, embedding_size)
    assert torch.is_tensor(output)
    # Check that output is not all zeros
    assert not torch.all(output == 0)