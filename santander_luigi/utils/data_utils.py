import pickle
import pandas as pd

from pandas.core.generic import NDFrame


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_pickle(path: str):
    with open(path, 'rb') as folds_file:
        obj: any = pickle.load(folds_file)
    return obj

def save_csv(data: NDFrame, path: str) -> pd.DataFrame:
    data.to_csv(path, index = False)
