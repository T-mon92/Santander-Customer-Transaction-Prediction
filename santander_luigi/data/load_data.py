from os.path import join
import numpy as np

import luigi

from santander_luigi.utils.data_utils import load_csv, save_csv
from santander_luigi.utils.pipeline_utils import posify

from santander_luigi.constants import STG1_PATH, DATA_PATH



class ExtractReal(luigi.Task):
    data_type = luigi.Parameter(default='train')

    def requires(self):
        return DataFile(self.data_type)

    def output(self):
        return luigi.LocalTarget(posify(STG1_PATH / f'{self.data_type}_real_samples.csv'))

    def run(self):
        data_req = self.input()

        data = load_csv(data_req.path)
        df_data = data.drop(['ID_code'], axis=1)
        df_data = df_data.values
        
        unique_samples = []
        unique_count = np.zeros_like(df_data)
        for feature in range(df_data.shape[1]):
            _, index_, count_ = np.unique(df_data[:, feature], return_counts=True, return_index=True)
            unique_count[index_[count_ == 1], feature] += 1

        real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
 
        del df_data

        data = data.iloc[real_samples_indexes]

        data_out = self.output()

        with self.output().open('w') as csv_file:
            data.to_csv(csv_file, index = False)


class DataFile(luigi.ExternalTask):
    data_type = luigi.Parameter(default='train')

    def output(self):
        return luigi.LocalTarget(posify(DATA_PATH / f'{self.data_type}.csv'))
