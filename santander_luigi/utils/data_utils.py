import pickle
import pandas as pd

from pandas.core.generic import NDFrame


def load_csv(path: str):
    return pd.read_csv(path)

def load_pickle(path: str):
    with open(path, 'rb') as folds_file:
        obj: any = pickle.load(folds_file)
    return obj

def save_csv(data: NDFrame, path: str):
    data.to_csv(path, index = False)

def save_pickle(obj: any, path: str):
    if not path.exists():
        path.parent.mkdir()
    else:
        pass
    with path.open('wb') as fs_output:
        pickle.dump(obj, fs_output, protocol=pickle.HIGHEST_PROTOCOL)