import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SRC.PositionalEncoding import PositionalEncoding
embedding_size = 16
seq_len = 10
batch_size = 4
pos_enc = PositionalEncoding(embedding_size, dropout=0.1, sequence_max_length=100)

# Create dummy input: shape (seq_len, batch_size, embedding_size)
dummy_input = torch.zeros(seq_len, batch_size, embedding_size)

# Pass through positional encoding
output = pos_enc(dummy_input)

# Print output info
print("Input shape: ", dummy_input.shape)
print("Output shape:", output.shape)
print("Output sample:\n", output[:2])  # print first two positions