import json
import numpy as np
import faiss
import os

def load_index_and_users():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "faiss_index.bin")
    user_ids_path = os.path.join(base_dir, "..", "vector_bank", "user_ids.json")
    
    index = faiss.read_index(index_path)
    with open(user_ids_path) as f:
        user_ids = json.load(f)
    return index, user_ids
