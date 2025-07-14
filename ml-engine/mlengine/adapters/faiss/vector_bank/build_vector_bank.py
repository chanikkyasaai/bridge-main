import numpy as np
import json
import os

output_dir = os.path.dirname(__file__)

users = ['user_A', 'user_B', 'user_C']
vectors_per_user = 10
dim = 64

all_vectors = []
all_ids = []

for user in users:
    center = np.random.rand(dim)
    for _ in range(vectors_per_user):
        noise = np.random.normal(0, 0.05, size=dim)
        vec = center + noise
        all_vectors.append(vec.astype("float32"))
        all_ids.append(user)

all_vectors = np.array(all_vectors).astype("float32")

# Save
np.save(os.path.join(output_dir, "user_vectors.npy"), all_vectors)
with open(os.path.join(output_dir, "user_ids.json"), "w") as f:
    json.dump(all_ids, f)

print("✅ User vectors and IDs saved.")
