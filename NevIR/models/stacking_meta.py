import numpy as np
from sklearn.linear_model import LogisticRegression

class StackingMetaLearner:
    def __init__(self, base_rankers: list):
        self.base = base_rankers
        self.clf  = LogisticRegression()

    def train(self, df):
        X, y = [], []
        for _, ex in df.iterrows():
            feats_q1 = []
            for r in self.base:
                s = r.score(ex.doc1, ex.doc2, ex.q1, ex.q2)
                feats_q1.extend(s["q1"])
            X.append(feats_q1)
            y.append(0)

            feats_q2 = []
            for r in self.base:
                s = r.score(ex.doc1, ex.doc2, ex.q1, ex.q2)
                feats_q2.extend(s["q2"])
            X.append(feats_q2)
            y.append(1)

        self.clf.fit(X, y)

    def score(self, doc1, doc2, q1, q2):
        feats_q1 = []
        for r in self.base:
            s = r.score(doc1, doc2, q1, q2)
            feats_q1.extend(s["q1"])
        proba_q1 = self.clf.predict_proba([feats_q1])[0]

        feats_q2 = []
        for r in self.base:
            s = r.score(doc1, doc2, q1, q2)
            feats_q2.extend(s["q2"])
        proba_q2 = self.clf.predict_proba([feats_q2])[0]

        return {"q1": proba_q1.tolist(), "q2": proba_q2.tolist()}
