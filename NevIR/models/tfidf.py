from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRanker:
    def __init__(self):
        self.vec = TfidfVectorizer()

    def fit(self, docs: list[str]):
        self.vec.fit(docs)

    def score(self, doc1, doc2, q1, q2):
        X = self.vec.transform([doc1, doc2, q1, q2])
        sims = cosine_similarity(X[2:], X[:2])
        return {"q1": sims[0].tolist(), "q2": sims[1].tolist()}

