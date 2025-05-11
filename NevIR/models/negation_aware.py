import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import Dataset
import torch

class NegationAwareRanker:
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _prepare_dataset(self, raw_split) -> Dataset:
        rows = []
        for ex in raw_split:
            rows.extend([
                {"query": ex["q1"], "document": ex["doc1"], "label": 1},
                {"query": ex["q1"], "document": ex["doc2"], "label": 0},
                {"query": ex["q2"], "document": ex["doc1"], "label": 0},
                {"query": ex["q2"], "document": ex["doc2"], "label": 1},
            ])
        ds = Dataset.from_list(rows)
        ds = ds.map(
            lambda batch: self.tokenizer(
                batch["query"], batch["document"],
                truncation=True, padding="max_length", max_length=128
            ),
            batched=True
        ).rename_column("label", "labels")
        return ds

    def train(self, train_split, val_split=None, output_dir="./negation_aware"):
        train_ds = self._prepare_dataset(train_split)
        eval_ds  = self._prepare_dataset(val_split) if val_split else None
        data_collator = DataCollatorWithPadding(self.tokenizer)
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            evaluation_strategy="steps" if val_split else "no",
            eval_steps=500,
            logging_steps=500,
            save_steps=1000,
            save_total_limit=2,
            learning_rate=2e-5,
            weight_decay=0.01,
        )
        def compute_metrics(p):
            preds = np.argmax(p.logits, axis=-1)
            return {"accuracy": (preds == p.label_ids).mean()}
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if val_split else None
        )
        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def score(self, doc1, doc2, q1, q2):
        def rel(q, d):
            enc = self.tokenizer(q, d, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            return logits.softmax(dim=-1)[0,1].item()
        return {"q1": [rel(q1, doc1), rel(q1, doc2)],
                "q2": [rel(q2, doc1), rel(q2, doc2)]}