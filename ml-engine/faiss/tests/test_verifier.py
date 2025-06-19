import numpy as np
import sys
import os

# Add the faiss directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_dir = os.path.join(script_dir, '..')
sys.path.insert(0, faiss_dir)

# Now import from the verifier module
from verifier.verify import verify_user

# Simulate an input vector close to user_B cluster
# Fix the path to the vector file
script_dir = os.path.dirname(os.path.abspath(__file__))
vector_path = os.path.join(script_dir, "..", "vector_bank", "user_vectors.npy")
center = np.load(vector_path)[10]  # pick user_B
v_current = center + np.random.normal(0, 0.04, size=64)

result = verify_user(v_current)
print("🧪 FAISS Verification Result:")
print(result)
