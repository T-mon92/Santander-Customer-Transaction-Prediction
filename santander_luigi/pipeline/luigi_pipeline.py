import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import luigi

from santander_luigi.utils.data_utils import load_csv, load_pickle
from santander_luigi.utils.pipeline_utils import posify
from santander_luigi.features.make_features import make_FE_features
from santander_luigi.train.train_models import train_catboost
from santander_luigi.validation.make_validation import create_folds

from santander_luigi.constants import STG1_PATH, STG2_PATH, STG3_PATH, RESULT_PATH, DATA_PATH


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

        folds_path = Path(self.output().path)

        with folds_path.open('wb') as fs_output:
            pickle.dump(folds, fs_output, protocol=pickle.HIGHEST_PROTOCOL)


class TrainFoldsModel(luigi.Task):

    def requires(self):
        return {
            'train': DataFile('train'),
            'data': GetRealExamples(),
            'folds': MakeFolds()
        }

    def output(self):
        return {
            'models': luigi.LocalTarget(posify(STG3_PATH / f'catboost_models.pickle')),
            'predictions': luigi.LocalTarget(posify(STG3_PATH / f'catboost_oof_preds.pickle')),
            'features': luigi.LocalTarget(posify(STG3_PATH / f'feature_list.pickle'))
        }

    def run(self):
        train_req = self.input()['train']
        data_req = self.input()['data']
        folds_req = self.input()['folds']

        train: pd.DataFrame = load_csv(train_req.path)
        data: pd.DataFrame = load_csv(data_req.path)
        folds: list = load_pickle(folds_req.path)

        features = train.drop(['ID_code', 'target'], axis=1).columns.tolist()

        oof = np.zeros(len(train))
        models = list()
        scores = list()

        for fold_, (trn_idx, val_idx) in enumerate(folds):
            X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']
            X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']

            X_train, X_valid = make_FE_features(X_train, data, features), make_FE_features(X_valid, data, features)

            clf, pred, score = train_catboost(X_train, y_train, X_valid, y_valid)

            oof[val_idx] = pred
            models.append(clf)
            scores.append(score)

        path_models = Path(self.output()['models'].path)
        path_predictions = Path(self.output()['predictions'].path)
        path_features = Path(self.output()['features'].path)

        with path_models.open('wb') as models_output:
            pickle.dump(models, models_output, protocol=pickle.HIGHEST_PROTOCOL)

        with path_predictions.open('wb') as preds_output:
            pickle.dump(oof, preds_output, protocol=pickle.HIGHEST_PROTOCOL)

        with path_features.open('wb') as feat_output:
            pickle.dump(features, feat_output, protocol=pickle.HIGHEST_PROTOCOL)


class GetSubmit(luigi.Task):

    def requires(self):
        return {'test_df': DataFile('test'),
                'trained_model': TrainFoldsModel()
                }

    def output(self):
        return luigi.LocalTarget(posify(RESULT_PATH / f'submission.csv'))

    def run(self):

        test_req = self.input()['test_df']
        models_req = self.input()['trained_model']['models']
        features_req = self.input()['trained_model']['features']

        test = load_csv(test_req.path)
        models = load_pickle(models_req.path)
        features = load_pickle(features_req.path)

        predictions = np.zeros(shape=(len(features), len(models)))

        for ind, model in enumerate(models):
            predictions[:, ind] = model.predict(test[features])

        predictions = np.mean(predictions, axis=1)

        sub = pd.DataFrame({"ID_code": test.ID_code.values})
        sub["target"] = predictions

        with self.output().open('w') as csv_file:
            sub.to_csv(csv_file, index = False)
