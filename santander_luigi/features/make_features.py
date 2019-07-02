import pandas as pd
from tqdm import tqdm

def make_FE_features(train_df : pd.DataFrame, valid_df : pd.DataFrame, test_df : pd.DataFrame,
                     whole_data_df : pd.DataFrame, feature_list : list):
    for col in tqdm(feature_list):
        gr = whole_data_df[col].value_counts()
        gr_bin = whole_data_df.groupby(col)[col].count() > 1

        train_df[col + '_un'] = train_df[col].map(gr).astype('category').cat.codes
        valid_df[col + '_un'] = valid_df[col].map(gr).astype('category').cat.codes
        test_df[col + '_un'] = test_df[col].map(gr).astype('category').cat.codes

        train_df[col + '_un_bin'] = train_df[col].map(gr_bin).astype('category').cat.codes
        valid_df[col + '_un_bin'] = valid_df[col].map(gr_bin).astype('category').cat.codes
        test_df[col + '_un_bin'] = test_df[col].map(gr_bin).astype('category').cat.codes

        train_df[col + '_raw_mul'] = train_df[col] * train_df[col + '_un_bin']
        valid_df[col + '_raw_mul'] = valid_df[col] * valid_df[col + '_un_bin']
        test_df[col + '_raw_mul'] = test_df[col] * test_df[col + '_un_bin']

        train_df[col + '_raw_mul_2'] = train_df[col] * train_df[col + '_un']
        valid_df[col + '_raw_mul_2'] = valid_df[col] * valid_df[col + '_un']
        test_df[col + '_raw_mul_2'] = test_df[col] * test_df[col + '_un']

        train_df[col + '_raw_mul_3'] = train_df[col + '_un_bin'] * train_df[col + '_un']
        valid_df[col + '_raw_mul_3'] = valid_df[col + '_un_bin'] * valid_df[col + '_un']
        test_df[col + '_raw_mul_3'] = test_df[col + '_un_bin'] * test_df[col + '_un']

    return train_df, valid_df, test_df