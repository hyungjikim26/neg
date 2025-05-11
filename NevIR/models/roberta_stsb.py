import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class STSBRegressorRanker:
    def __init__(self, model_dir: str = "models/stsb_roberta_regressor/checkpoint-1800", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def score(self, doc1, doc2, q1, q2):
        def run(q, d):
            enc = self.tokenizer(
                q, d,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                return self.model(**enc).logits.squeeze(-1).item()

        s11 = run(q1, doc1)
        s12 = run(q1, doc2)
        s21 = run(q2, doc1)
        s22 = run(q2, doc2)
        return {"q1": [s11, s12], "q2": [s21, s22]}