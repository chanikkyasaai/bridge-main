import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SRC.UserBehaviorTransformer import UserBehaviorTransformer
from SRC.InputEmbedding import InputEmbeddings
from SRC.PositionalEncoding import PositionalEncoding

# --- Your Model Code Should Be Defined Here ---
# (Assumes InputEmbeddings, PositionalEncoding, and UserBehaviorTransformer are already defined)

# --- Testing Config ---
action_vocab_size = 100     # number of unique actions (tokens)
embedding_size = 64         # dimension of embedding vector
num_heads = 4               # number of attention heads
num_layers = 2              # transformer layers
seq_len = 20                # length of each input sequence
batch_size = 8              # number of samples per batch

# --- Instantiate Model ---
model = UserBehaviorTransformer(
    action_vocab_size=action_vocab_size,
    embedding_size=embedding_size,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_dim=128,
    dropout=0.1,
    max_seq_len=500
)

# --- Dummy Input ---
dummy_input = torch.randint(0, action_vocab_size, (batch_size, seq_len))  # shape: (batch, seq_len)

# --- Forward Pass ---
output_logits = model(dummy_input)  # shape: (batch_size, 2)

# --- Convert to Class Probabilities and Predictions ---
probabilities = torch.softmax(output_logits, dim=1)
predictions = torch.argmax(probabilities, dim=1)

# --- Print Results ---
print("Output logits shape:", output_logits.shape)
print("Probabilities:", probabilities)
print("Predicted class indices:", predictions)
