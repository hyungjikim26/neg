from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class BiEncoderRanker:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, doc1, doc2, q1, q2):
        embs_q = self.model.encode([q1, q2])
        embs_d = self.model.encode([doc1, doc2])
        sims = cosine_similarity(embs_q, embs_d)
        return {"q1": sims[0].tolist(), "q2": sims[1].tolist()}
