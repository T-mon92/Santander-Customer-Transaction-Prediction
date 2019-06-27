import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


def create_folds(data: pd.DataFrame):
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(data, y=np.zeros(len(data))))
