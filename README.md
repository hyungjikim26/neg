# Negation-Aware Ranking on the NevIR Dataset

This project explores a variety of information retrieval models designed to handle negation, using the [NevIR benchmark dataset](https://huggingface.co/datasets/orionweller/NevIR). Models include TF-IDF, transformer-based encoders (with and without negation cue marking), semantic pretraining on STS-B, and pairwise fine-tuning.

The best-performing model—a RoBERTa-large model pretrained on STS-B and fine-tuned on NevIR—achieves a pairwise accuracy of 0.797.

## How to Run Evaluation

Clone the repository run:

```bash
python evaluate.py
```
