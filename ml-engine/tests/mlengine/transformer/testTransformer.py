import torch
import sys
import os
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mlengine.adapters.transformer.SRC.UserBehaviorTransformer import UserBehaviorTransformer
from mlengine.adapters.transformer.SRC.InputEmbedding import InputEmbeddings
from mlengine.adapters.transformer.SRC.PositionalEncoding import PositionalEncoding

@pytest.mark.parametrize("batch_size, seq_len", [(8, 20), (4, 10)])
def test_transformer_forward(batch_size, seq_len):
    action_vocab_size = 100
    embedding_size = 64
    num_heads = 4
    num_layers = 2
    model = UserBehaviorTransformer(
        action_vocab_size=action_vocab_size,
        embedding_size=embedding_size,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=128,
        dropout=0.1,
        max_seq_len=500
    )
    dummy_input = torch.randint(0, action_vocab_size, (batch_size, seq_len))
    output_logits = model(dummy_input)
    assert output_logits.shape == (batch_size, 2)
    probabilities = torch.softmax(output_logits, dim=1)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    predictions = torch.argmax(probabilities, dim=1)
    assert predictions.shape == (batch_size,)
