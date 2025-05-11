import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import mean_squared_error


from scipy.stats import pearsonr, spearmanr

MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./stsb_roberta_regressor"
BATCH_SIZE = 32
EPOCHS     = 10
MAX_LEN    = 128

def compute_metrics(pred):
    preds = pred.predictions.squeeze(-1)
    labels = pred.label_ids
    mse = mean_squared_error(labels, preds)
    pearson_corr = pearsonr(labels, preds)[0]
    spearman_corr = spearmanr(labels, preds)[0]
    return {
        "mse": mse,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
    }

def main():
    ds = load_dataset("glue", "stsb")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )

    ds = ds.map(preprocess, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=1,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()