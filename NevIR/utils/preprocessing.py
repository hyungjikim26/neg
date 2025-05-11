import json
import re


def load_cue_pattern(cue_file: str = "cues.json") -> re.Pattern:
    with open(cue_file, 'r') as f:
        cues = json.load(f)["cues"]
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(c) for c in cues) + r")\b",
        flags=re.IGNORECASE
    )
    return pattern


def mark_cues(text: str, pattern: re.Pattern) -> str:

    return pattern.sub(r"[NEG] \g<0> [NEG]", text)


class PreprocessingRanker:
    def __init__(self, base_ranker, cue_pattern: re.Pattern):
        self.base = base_ranker
        self.pattern = cue_pattern

    def score(self, doc1, doc2, q1, q2):
        d1 = mark_cues(doc1, self.pattern)
        d2 = mark_cues(doc2, self.pattern)
        qq1 = mark_cues(q1,    self.pattern)
        qq2 = mark_cues(q2,    self.pattern)
        return self.base.score(d1, d2, qq1, qq2)