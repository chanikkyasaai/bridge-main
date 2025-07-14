import numpy as np

class TransformerBehavioralComparator:
    def __init__(self):
        # Simulated transformer configuration
        self.embedding_dim = 64

    def embed_sequence(self, sequence):
        """
        Simulates transformer embedding over a behavioral vector sequence.
        Returns the averaged embedding vector.
        """
        if len(sequence) == 0:
            return np.zeros(self.embedding_dim)

        sequence = np.array(sequence).astype("float32")
        return np.mean(sequence, axis=0)

    def similarity_score(self, live_embedding, reference_embedding):
        """
        Cosine similarity between two embeddings.
        """
        live = live_embedding / np.linalg.norm(live_embedding)
        ref = reference_embedding / np.linalg.norm(reference_embedding)
        return float(np.dot(live, ref))

    def compare(self, live_sequence, reference_sequences, context_factors):
        """
        Compares a live session's behavior with past reference embeddings
        and adjusts based on context.
        """
        live_emb = self.embed_sequence(live_sequence)

        similarities = []
        for ref_seq in reference_sequences:
            ref_emb = self.embed_sequence(ref_seq)
            score = self.similarity_score(live_emb, ref_emb)
            similarities.append(score)

        base_score = float(np.mean(similarities))

        # Adjust score based on context
        penalty = 0
        if context_factors.get("age_group") == "elderly":
            penalty -= 0.05
        if context_factors.get("time_of_day") in ["late_night"]:
            penalty -= 0.04
        if context_factors.get("device_type") in ["unknown", "new"]:
            penalty -= 0.03

        final_score = base_score + penalty
        return max(0.0, min(1.0, final_score))  # clamp between 0 and 1
