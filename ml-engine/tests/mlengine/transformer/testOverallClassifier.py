import sys
import os
import torch
import pytest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from torch.utils.data import TensorDataset, random_split
from mlengine.adapters.transformer.SRC.UserBehaviorClassifier import UserBehaviorClassifier
from mlengine.adapters.transformer.SRC.UserBehaviorTransformer import UserBehaviorTransformer

@pytest.mark.parametrize("num_samples, seq_len, vocab_size", [(100, 10, 50)])
def test_overall_classifier(num_samples, seq_len, vocab_size):
    test_split = 0.2
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, 2, (num_samples,))
    full_dataset = TensorDataset(X, y)
    test_size = int(num_samples * test_split)
    train_size = num_samples - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    model = UserBehaviorTransformer(vocab_size, 32, 2, 1)
    clf = UserBehaviorClassifier(model)
    clf.train(train_dataset, epochs=1)
    acc = clf.test(test_dataset)
    assert 0.0 <= acc <= 1.0
    sample = torch.randint(0, vocab_size, (1, seq_len))
    pred, prob = clf.predict(sample)
    assert pred in [0, 1]
    assert 0.0 <= prob <= 1.0
