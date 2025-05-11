import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

SEED       = 42
BATCH_SIZE = 16
EPOCHS     = 8
LR         = 2e-5
MAX_LEN    = 512
STS_CHECKPT= "./stsb_roberta_regressor_large/checkpoint-900"
OUT_DIR    = "./roberta_nevir_pairwise_large"

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.raw = split
        self.tokenizer = AutoTokenizer.from_pretrained(STS_CHECKPT)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        example = self.raw[idx]
        pairs = [
            (example["q1"], example["doc1"], example["doc2"]),
            (example["q2"], example["doc2"], example["doc1"])
        ]
        data = []
        for q, d1, d2 in pairs:
            if random.random() < 0.5:
                a, b, label = d1, d2, 0
            else:
                a, b, label = d2, d1, 1
            ex1 = self.tokenizer(
                q, a,
                truncation=True, padding="max_length", max_length=MAX_LEN,
                return_tensors="pt",
            )
            ex2 = self.tokenizer(
                q, b,
                truncation=True, padding="max_length", max_length=MAX_LEN,
                return_tensors="pt",
            )
            data.append({
                "input_ids1": ex1.input_ids.squeeze(0),
                "attn_mask1": ex1.attention_mask.squeeze(0),
                "input_ids2": ex2.input_ids.squeeze(0),
                "attn_mask2": ex2.attention_mask.squeeze(0),
                "label": torch.tensor(label, dtype=torch.long),
            })
        return data

def collate_fn(batch):
    flat = [item for sub in batch for item in sub]
    def stack(key):
        return torch.stack([d[key] for d in flat])
    return {
        "input_ids1": stack("input_ids1"),
        "attn_mask1": stack("attn_mask1"),
        "input_ids2": stack("input_ids2"),
        "attn_mask2": stack("attn_mask2"),
        "labels":      stack("label"),
    }

def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = load_dataset("orionweller/NevIR")
    train_ds = PairwiseDataset(raw["train"])
    val_ds   = PairwiseDataset(raw["validation"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn)

    model = AutoModelForSequenceClassification.from_pretrained(
        STS_CHECKPT, num_labels=1, ignore_mismatched_sizes=True
    )
    model.config.problem_type = "regression"
    model.to(device)

    optim = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=100, num_training_steps=total_steps
    )

    best_val = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Train {epoch}"):
            batch = {k:v.to(device) for k,v in batch.items()}
            out1 = model(
                input_ids=batch["input_ids1"],
                attention_mask=batch["attn_mask1"],
            ).logits.squeeze(-1)
            out2 = model(
                input_ids=batch["input_ids2"],
                attention_mask=batch["attn_mask2"],
            ).logits.squeeze(-1)
            logits = torch.stack([out1, out2], dim=1)
            loss   = F.cross_entropy(logits, batch["labels"])
            loss.backward()
            optim.step(); scheduler.step(); optim.zero_grad()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                batch = {k:v.to(device) for k,v in batch.items()}
                out1 = model(
                    input_ids=batch["input_ids1"],
                    attention_mask=batch["attn_mask1"],
                ).logits.squeeze(-1)
                out2 = model(
                    input_ids=batch["input_ids2"],
                    attention_mask=batch["attn_mask2"],
                ).logits.squeeze(-1)
                preds = torch.argmax(torch.stack([out1, out2], dim=1), dim=1)
                correct += (preds == batch["labels"]).sum().item()
                total   += preds.size(0)
        acc = correct / total
        print(f"Epoch {epoch} — val pair accuracy: {acc:.3f}")
        
        if acc > best_val:
            best_val = acc
            model.save_pretrained(OUT_DIR)
            print(f"→ saved checkpoint (best={best_val:.3f})")

if __name__ == "__main__":
    train()