import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import Pool, CatBoostClassifier
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import luigi

from santander_luigi.utils.data_utils import load_csv
from santander_luigi.utils.pipeline_utils import posify

from santander_luigi.constants import STG1_PATH, RESULT_PATH, DATA_PATH


class DataFile(luigi.ExternalTask):
    data_type = luigi.Parameter(default='train')

    def output(self):
        return luigi.LocalTarget(posify(DATA_PATH / f'{self.data_type}.csv'))


class GetRealExamples(luigi.Task):
    def requires(self):
        return (DataFile('train'), DataFile('test'))

    def output(self):
        return luigi.LocalTarget(posify(STG1_PATH / f'data.csv'))

    def run(self):
        train = load_csv(self.input()[0].path)
        test = load_csv(self.input()[1].path)

        df_data = test.drop(['ID_code'], axis=1)
        df_data = df_data.values

        unique_count = np.zeros_like(df_data)
        for feature in range(df_data.shape[1]):
            _, index_, count_ = np.unique(df_data[:, feature], return_counts=True, return_index=True)
            unique_count[index_[count_ == 1], feature] += 1

        real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]

        del df_data

        test_real = test.iloc[real_samples_indexes]


        data = train.append(test_real)

        with self.output().open('w') as csv_file:
            data.to_csv(csv_file, index=False)



class ExtractReal(luigi.Task):
    data_type = luigi.Parameter(default='train')

    def requires(self):
        return (DataFile('train'), DataFile('test'), GetRealExamples())

    def output(self):
        return luigi.LocalTarget(posify(RESULT_PATH / f'submission.csv'))

    def run(self):

        train = load_csv(self.input()[0].path)
        test = load_csv(self.input()[1].path)
        data = load_csv(self.input()[2].path)

        features = train.drop(['ID_code', 'target'], axis=1).columns.tolist()

        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(train))
        predictions = np.zeros(len(test))

        model = CatBoostClassifier(loss_function="Logloss",
                                   eval_metric="AUC",
                                   task_type="GPU",
                                   learning_rate=0.01,
                                   iterations=70000,
                                   l2_leaf_reg=50,
                                   random_seed=42,
                                   od_type="Iter",
                                   depth=5,
                                   early_stopping_rounds=15000,
                                   border_count=64
                                   )

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, train.target.values)):
            print("Fold {}".format(fold_))
            X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
            X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']

            for col in tqdm(features):
                gr = data[col].value_counts()
                gr_bin = data.groupby(col)[col].count() > 1

                X_train[col + '_un'] = X_train[col].map(gr).astype('category').cat.codes
                X_valid[col + '_un'] = X_valid[col].map(gr).astype('category').cat.codes
                test[col + '_un'] = test[col].map(gr).astype('category').cat.codes

                X_train[col + '_un_bin'] = X_train[col].map(gr_bin).astype('category').cat.codes
                X_valid[col + '_un_bin'] = X_valid[col].map(gr_bin).astype('category').cat.codes
                test[col + '_un_bin'] = test[col].map(gr_bin).astype('category').cat.codes

                X_train[col + '_raw_mul'] = X_train[col] * X_train[col + '_un_bin']
                X_valid[col + '_raw_mul'] = X_valid[col] * X_valid[col + '_un_bin']
                test[col + '_raw_mul'] = test[col] * test[col + '_un_bin']

                X_train[col + '_raw_mul_2'] = X_train[col] * X_train[col + '_un']
                X_valid[col + '_raw_mul_2'] = X_valid[col] * X_valid[col + '_un']
                test[col + '_raw_mul_2'] = test[col] * test[col + '_un']

                X_train[col + '_raw_mul_3'] = X_train[col + '_un_bin'] * X_train[col + '_un']
                X_valid[col + '_raw_mul_3'] = X_valid[col + '_un_bin'] * X_valid[col + '_un']
                test[col + '_raw_mul_3'] = test[col + '_un_bin'] * test[col + '_un']

            _train = Pool(X_train, label=y_train)
            _valid = Pool(X_valid, label=y_valid)
            clf = model.fit(_train,
                            eval_set=_valid,
                            use_best_model=True,
                            verbose=5000)
            pred = clf.predict_proba(X_valid)[:, 1]
            oof[val_idx] = pred
            print("  auc = ", roc_auc_score(y_valid, pred))
            predictions += clf.predict_proba(test.drop('ID_code', axis=1))[:, 1] / folds.n_splits
        print("CV score: {:<8.5f}".format(roc_auc_score(train.target, oof)))

        sub = pd.DataFrame({"ID_code": test.ID_code.values})
        sub["target"] = predictions


        with self.output().open('w') as csv_file:
            sub.to_csv(csv_file, index = False)
