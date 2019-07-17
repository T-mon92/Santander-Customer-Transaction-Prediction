import pandas as pd
from tqdm import tqdm

def make_FE_features(df_train: pd.DataFrame,
                     df_valid: pd.DataFrame,
                     df_test: pd.DataFrame,
                     whole_data_df : pd.DataFrame,
                     feature_list : list):
    for col in tqdm(feature_list):
        gr = whole_data_df[col].value_counts()
        gr_bin = whole_data_df.groupby(col)[col].count() > 1

        df_train[col + '_un'] = df_train[col].map(gr).astype('category').cat.codes
        df_valid[col + '_un'] = df_valid[col].map(gr).astype('category').cat.codes
        df_test[col + '_un'] = df_test[col].map(gr).astype('category').cat.codes

        df_train[col + '_un_bin'] = df_train[col].map(gr_bin).astype('category').cat.codes
        df_valid[col + '_un_bin'] = df_valid[col].map(gr_bin).astype('category').cat.codes
        df_test[col + '_un_bin'] = df_test[col].map(gr_bin).astype('category').cat.codes

        df_train[col + '_raw_mul'] = df_train[col] * df_train[col + '_un_bin']
        df_valid[col + '_raw_mul'] = df_valid[col] * df_valid[col + '_un_bin']
        df_test[col + '_raw_mul'] = df_test[col] * df_test[col + '_un_bin']

        df_train[col + '_raw_mul_2'] = df_train[col] * df_train[col + '_un']
        df_valid[col + '_raw_mul_2'] = df_valid[col] * df_valid[col + '_un']
        df_test[col + '_raw_mul_2'] = df_test[col] * df_test[col + '_un']

        df_train[col + '_raw_mul_3'] = df_train[col + '_un_bin'] * df_train[col + '_un']
        df_valid[col + '_raw_mul_3'] = df_valid[col + '_un_bin'] * df_valid[col + '_un']
        df_test[col + '_raw_mul_3'] = df_test[col + '_un_bin'] * df_test[col + '_un']

    return df_train, df_valid, df_test
