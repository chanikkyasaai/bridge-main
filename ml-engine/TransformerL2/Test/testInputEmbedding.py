import torch
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SRC.InputEmbedding import InputEmbeddings

action_count = 10         # Suppose you have 10 different actions
embedding_size = 4        # Each action will be embedded into a 4D vector

# Instantiate the model
embedding_model = InputEmbeddings(action_count, embedding_size)

# Create dummy input (batch_size=2, seq_len=3) of action indices
dummy_input = torch.LongTensor([[1, 3, 5], [2, 4, 6]])

# Pass input through model
output = embedding_model(dummy_input)

# Print output
print("Output shape:", output.shape)  # Should be [2, 3, 4]
print("Output dtype:", output.dtype)  # Should be torch.float32
print("Output vectors: ",output)