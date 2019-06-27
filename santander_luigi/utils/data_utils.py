import pandas as pd
from pandas.core.generic import NDFrame


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(data: NDFrame, path: str, key: str) -> pd.DataFrame:
    data.to_csv(path, index = False)
