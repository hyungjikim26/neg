from models.negation_aware import NegationAwareRanker

class CueAwareRanker(NegationAwareRanker):
    def __init__(self, model_name="bert-base-uncased", cue_file="cues.json", device=None):
        super().__init__(model_name, device)
        import json, re
        cues = json.load(open(cue_file))["cues"]
        self.pattern = re.compile(
            r"\b(?:" + "|".join(map(re.escape, cues)) + r")\b",
            flags=re.IGNORECASE
        )

    def _mark(self, text: str) -> str:
        return self.pattern.sub(r"[NEG] \g<0> [NEG]", text)

    def score(self, doc1, doc2, q1, q2):
        return super().score(
            self._mark(doc1), self._mark(doc2),
            self._mark(q1),    self._mark(q2)
        )
