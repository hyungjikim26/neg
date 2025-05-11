import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RegressionCrossEncoder:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def score(self, doc1, doc2, q1, q2):
        pairs = [(q1, doc1), (q1, doc2), (q2, doc1), (q2, doc2)]
        enc = self.tokenizer(
            [q for q,d in pairs], [d for q,d in pairs],
            padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits.squeeze(-1)
        sims = logits.view(2,2).cpu().tolist()
        return {"q1": sims[0], "q2": sims[1]}
