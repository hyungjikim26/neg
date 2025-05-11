import pandas as pd
from datasets import load_dataset

def load_nevir(split: str = 'train') -> pd.DataFrame:
    ds = load_dataset("orionweller/NevIR", split=split)
    return pd.DataFrame(ds)