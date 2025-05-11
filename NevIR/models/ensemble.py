class EnsembleRanker:
    def __init__(self, rankers: list, weights: list[float]):
        assert len(rankers) == len(weights)
        total = sum(weights)
        self.rankers = rankers
        self.weights = [w/total for w in weights]

    def score(self, doc1, doc2, q1, q2):
        agg = {"q1": [0.0, 0.0], "q2": [0.0, 0.0]}
        for r, w in zip(self.rankers, self.weights):
            s = r.score(doc1, doc2, q1, q2)
            agg["q1"][0] += w * s["q1"][0]
            agg["q1"][1] += w * s["q1"][1]
            agg["q2"][0] += w * s["q2"][0]
            agg["q2"][1] += w * s["q2"][1]
        return agg