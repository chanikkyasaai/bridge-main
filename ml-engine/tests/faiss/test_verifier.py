import numpy as np
import sys
import os
import pytest

# Add the faiss directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_dir = os.path.join(script_dir, '../../')
sys.path.insert(0, faiss_dir)

from mlengine.adapters.faiss.verifier.verify import verify_user

pytestmark = pytest.mark.skipif(sys.platform.startswith('win'), reason='FAISS is not fully supported on Windows')

@pytest.mark.parametrize("user_idx", [10, 5, 0])
def test_faiss_verifier(user_idx):
    vector_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../adapters/faiss/vector_bank/user_vectors.npy'))
    vectors = np.load(vector_path)
    center = vectors[user_idx]
    v_current = center + np.random.normal(0, 0.04, size=64)
    result = verify_user(v_current)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"decision", "matched_user", "match_count", "avg_score"}
    assert result["decision"] in ["continue", "escalate", "block"]
    assert isinstance(result["matched_user"], (str, int))
    assert isinstance(result["match_count"], int)
    assert isinstance(result["avg_score"], float)
