import re
import json
import numpy as np
import pandas as pd

from process_data import load_nevir
from utils.preprocessing import load_cue_pattern, mark_cues, PreprocessingRanker

from models.random_ranker      import RandomRanker
from models.tfidf              import TfidfRanker
from models.cross_encoder      import CrossEncoderRanker
from models.negation_aware     import NegationAwareRanker
from models.cue_aware          import CueAwareRanker
from models.bi_encoder         import BiEncoderRanker
from models.regression_encoder import RegressionCrossEncoder
from models.ensemble           import EnsembleRanker
from models.stacking_meta      import StackingMetaLearner

from models.roberta_stsb       import STSBRegressorRanker
from models.roberta_nevir      import RoBERTaNevirRanker
from models.roberta_nevir_v2   import RoBERTaNevirRankerV2
from models.roberta_nevir_large import RoBERTaNevirRankerLarge

def evaluate(df: pd.DataFrame, ranker, name: str):
    y1, y2, p1, p2 = [], [], [], []
    for _, row in df.iterrows():
        scores = ranker.score(row.doc1, row.doc2, row.q1, row.q2)
        y1.append(0);  y2.append(1)
        p1.append(int(np.argmax(scores["q1"])))
        p2.append(int(np.argmax(scores["q2"])))

    acc1 = (np.array(p1) == y1).mean()
    acc2 = (np.array(p2) == y2).mean()
    acc_pair = np.mean([(p1[i]==y1[i] and p2[i]==y2[i]) for i in range(len(y1))])

    print(f"--- {name} ---")
    print(f"Q1 acc: {acc1:.3f}   Q2 acc: {acc2:.3f}")
    print(f"Avg per-query acc: {(acc1+acc2)/2:.3f}")
    print(f"Per-pair acc:      {acc_pair:.3f}")
    print()


def evaluate(df: pd.DataFrame, ranker, name: str):
    p1 = []       
    p2 = []      
    y1 = []      
    y2 = []     
    pair_scores = []

    for _, row in df.iterrows():
        out   = ranker.score(row.doc1, row.doc2, row.q1, row.q2)
        pred1 = int(np.argmax(out["q1"]))
        pred2 = int(np.argmax(out["q2"]))

        y1.append(0);   y2.append(1)
        p1.append(pred1)
        p2.append(pred2)

        pair_scores.append(1 if (pred1 == 0 and pred2 == 1) else 0)

    acc1 = (np.array(p1) == y1).mean()
    acc2 = (np.array(p2) == y2).mean()
    avg_query = (acc1 + acc2) / 2

    pair_acc = np.mean(pair_scores)

    print(f"--- {name} ---")
    print(f"Q1 acc:          {acc1:.3f}")
    print(f"Q2 acc:          {acc2:.3f}")
    print(f"Avg per-query:   {avg_query:.3f}")
    print(f"Per-pair acc:    {pair_acc:.3f}")
    print(f"NevIR “score”:   {pair_acc:.3f}")
    print()

if __name__ == "__main__":
    train_df = load_nevir("train")
    val_df   = load_nevir("validation")
    test_df  = load_nevir("test")

    tfidf = TfidfRanker()
    tfidf.fit(
        train_df.doc1.tolist() +
        train_df.doc2.tolist()
    )

    cue_pattern = load_cue_pattern("cues.json")
    tfidf_cue = PreprocessingRanker(TfidfRanker(), cue_pattern)

    marked_docs = []
    for d in train_df.doc1.tolist() + train_df.doc2.tolist():
        marked_docs.append(mark_cues(d, cue_pattern))
    tfidf_cue.base.fit(marked_docs)

    models = {
        "random":      RandomRanker(seed=42),
        "tfidf":       tfidf,
        "tfidf_cue":   tfidf_cue,
        "cross":       CrossEncoderRanker(),
        "cross_cue":   PreprocessingRanker(CrossEncoderRanker(), cue_pattern),
        "neg":         NegationAwareRanker(),
        "neg_cue":     PreprocessingRanker(NegationAwareRanker(), cue_pattern),
        "bi":          BiEncoderRanker(),
        "bi_cue":      PreprocessingRanker(BiEncoderRanker(), cue_pattern),
        "reg":         RegressionCrossEncoder(),
        "reg_cue":     PreprocessingRanker(RegressionCrossEncoder(), cue_pattern),
    }

    base_list = [models[k] for k in ["tfidf", "cross", "neg"]]
    stack = StackingMetaLearner(base_list)
    stack.train(val_df)
    models["stack"] = stack
    ens = EnsembleRanker(list(models.values()), [1]*len(models))
    models["ensemble"] = ens

    models["stsb_roberta"] = STSBRegressorRanker()

    # fine-tuned model
    models["nevir_roberta"] = RoBERTaNevirRanker()
    models["nevir_roberta_v2"] = RoBERTaNevirRankerV2()
    models["nevir_roberta_large"] = RoBERTaNevirRankerLarge()


    for name, m in models.items():
        evaluate(test_df, m, name)

    