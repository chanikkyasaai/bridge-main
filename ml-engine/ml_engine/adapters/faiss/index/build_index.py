import numpy as np
import faiss
import os

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to the vector_bank directory
vecs_path = os.path.join(script_dir, "..", "vector_bank", "user_vectors.npy")
index_path = os.path.join(script_dir, "faiss_index.bin")

vectors = np.load(vecs_path).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(vectors)

index = faiss.IndexFlatIP(vectors.shape[1])  # Inner product = cosine sim
index.add(vectors)
faiss.write_index(index, index_path)

print("✅ FAISS index built and saved.")
