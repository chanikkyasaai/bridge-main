import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import TensorDataset, random_split
from SRC.UserBehaviorClassifier import UserBehaviorClassifier
from SRC.UserBehaviorTransformer import UserBehaviorTransformer
# --- Parameters ---
num_samples = 1000      # total samples
seq_len = 10            # sequence length
vocab_size = 100        # number of possible actions
test_split = 0.2        # 20% for testing

# --- Generate Random Dataset ---
X = torch.randint(0, vocab_size, (num_samples, seq_len))  # (samples, seq_len)
y = torch.randint(0, 2, (num_samples,))                   # binary labels (0 or 1)
print(X)
# --- Create TensorDataset ---
full_dataset = TensorDataset(X, y)

# --- Split into Train/Test ---
test_size = int(num_samples * test_split)
train_size = num_samples - test_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


model = UserBehaviorTransformer(100, 64, 4, 2)
clf = UserBehaviorClassifier(model)


clf.train(train_dataset, epochs=5)
clf.test(test_dataset)

# Predict single sample or batch
sample = torch.randint(0, 100, (1, 10))  # (batch_size=1, seq_len=10)
pred, prob = clf.predict(sample)
