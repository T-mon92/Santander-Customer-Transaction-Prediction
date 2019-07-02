import pandas as pd
from tqdm import tqdm

def make_FE_features(df : pd.DataFrame, whole_data_df : pd.DataFrame, feature_list : list):
    for col in tqdm(feature_list):
        gr = whole_data_df[col].value_counts()
        gr_bin = whole_data_df.groupby(col)[col].count() > 1

        df[col + '_un'] = df[col].map(gr).astype('category').cat.codes

        df[col + '_un_bin'] = df[col].map(gr_bin).astype('category').cat.codes

        df[col + '_raw_mul'] = df[col] * df[col + '_un_bin']

        df[col + '_raw_mul_2'] = df[col] * df[col + '_un']

        df[col + '_raw_mul_3'] = df[col + '_un_bin'] * df[col + '_un']

    return df
