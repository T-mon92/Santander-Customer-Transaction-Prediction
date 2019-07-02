import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle
from pathlib import Path

import luigi

from santander_luigi.utils.data_utils import load_csv, load_pickle
from santander_luigi.utils.pipeline_utils import posify
from santander_luigi.features.make_features import make_FE_features
from santander_luigi.train.train_models import train_catboost
from santander_luigi.validation.make_validation import create_folds

from santander_luigi.constants import STG1_PATH, STG2_PATH, RESULT_PATH, DATA_PATH


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
        samples_req = self.input()
        train, test = load_csv(samples_req[0].path), load_csv(samples_req[1].path)

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


class MakeFolds(luigi.Task):

    def requires(self):
        return DataFile('train')

    def output(self):
        return luigi.LocalTarget(posify(STG2_PATH / 'folds.pickle'))

    def run(self):

        data: pd.DataFrame = load_csv(self.input().path)
        folds = create_folds(data)

        path = Path(self.output().path)

        with path.open('wb') as fs_output:
            pickle.dump(folds, fs_output, protocol=pickle.HIGHEST_PROTOCOL)



class GetSubmit(luigi.Task):

    def requires(self):
        return (DataFile('train'), DataFile('test'), GetRealExamples(), MakeFolds())

    def output(self):
        return luigi.LocalTarget(posify(RESULT_PATH / f'submission.csv'))

    def run(self):

        submit_req = self.input()
        train, test, data = load_csv(submit_req[0].path), load_csv(submit_req[1].path), load_csv(submit_req[2].path)

        folds = load_pickle(submit_req[3].path)

        features = train.drop(['ID_code', 'target'], axis=1).columns.tolist()

        oof = np.zeros(len(train))
        predictions = np.zeros(len(test))


        for fold_, (trn_idx, val_idx) in enumerate(folds):
            print("Fold {}".format(fold_))
            X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
            X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']

            X_train, X_valid, test = make_FE_features(X_train, X_valid, test, data, features)

            clf, pred, score = train_catboost(X_train, y_train, X_valid, y_valid)

            oof[val_idx] = pred
            print("  auc = ", score)
            predictions += clf.predict_proba(test.drop('ID_code', axis=1))[:, 1] / folds.n_splits
        print("CV score: {:<8.5f}".format(roc_auc_score(train.target, oof)))

        sub = pd.DataFrame({"ID_code": test.ID_code.values})
        sub["target"] = predictions


        with self.output().open('w') as csv_file:
            sub.to_csv(csv_file, index = False)
