import numpy as np
import sys
import os
import faiss

# Add the faiss directory to the path for importing from other modules
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_dir = os.path.join(script_dir, '..')
sys.path.insert(0, faiss_dir)

from index.load_index import load_index_and_users

def verify_user(v_current: np.ndarray, k: int = 10):
    index, user_ids = load_index_and_users()

    # Normalize input vector
    v_current = v_current.astype("float32").reshape(1, -1)
    faiss.normalize_L2(v_current)

    D, I = index.search(v_current, k)
    I = I[0]  # indices
    D = D[0]  # similarity scores

    match_counts = {}
    for idx in I:
        user = user_ids[idx]
        match_counts[user] = match_counts.get(user, 0) + 1

    top_user = max(match_counts, key=match_counts.get)
    top_count = match_counts[top_user]
    top_score = np.mean(D[:top_count]) * 100

    if top_count >= 8 and top_score >= 90:
        decision = "continue"
    elif top_count >= 5:
        decision = "escalate"
    else:
        decision = "block"

    return {
        "decision": decision,
        "matched_user": top_user,
        "match_count": top_count,
        "avg_score": round(top_score, 2)
    }
