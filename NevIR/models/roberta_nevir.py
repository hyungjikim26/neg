import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RoBERTaNevirRanker:
    def __init__(self,
                 model_dir: str = "models/roberta_nevir_pairwise",
                 device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=1, ignore_mismatched_sizes=True
        )
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def score(self, doc1, doc2, q1, q2):
        def get_score(q, d):
            enc = self.tokenizer(
                q, d,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            return self.model(**enc).logits.squeeze(-1).item()

        s11 = get_score(q1, doc1)
        s12 = get_score(q1, doc2)
        s21 = get_score(q2, doc1)
        s22 = get_score(q2, doc2)
        return {"q1": [s11, s12], "q2": [s21, s22]}