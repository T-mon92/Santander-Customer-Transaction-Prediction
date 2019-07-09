import pickle
import pandas as pd
import numpy as np

from pandas.core.generic import NDFrame

def reduce_memory_usage(data: pd.DataFrame) -> pd.DataFrame:
    types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = data.memory_usage().sum() / 1024 ** 2
    for col in data.columns:
        col_type = data[col].dtypes
        if col_type in types:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024 ** 2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {(100 * (start_mem - end_mem) / start_mem):.1f}%')
    return data

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
        path.parent.mkdir(exist_ok=True)
    with path.open('wb') as fs_output:
        pickle.dump(obj, fs_output, protocol=pickle.HIGHEST_PROTOCOL)