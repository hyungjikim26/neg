import random

class RandomRanker:
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

    def score(self, doc1, doc2, q1, q2):
        scores = {}
        for idx in (1, 2):
            if random.random() > 0.5:
                scores[f"q{idx}"] = [1, 0]
            else:
                scores[f"q{idx}"] = [0, 1]
        return scores
