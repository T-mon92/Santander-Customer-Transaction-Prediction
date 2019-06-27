from os.path import join

import luigi

from santander_luigi.utils.data_utils import load_csv, save_csv
from santander_luigi.utils.pipeline_utils import posify

from santander_luigi.constants import STG1_PATH, DATA_PATH



class LoadData(luigi.Task):
    data_type = luigi.Parameter(default='train')

    def requires(self):
        return DataFile(self.data_type)

    def run(self):
        data_req = self.input()

        data = load_csv(data_req.path)

        data_out = self.output()

        save_csv(data, data_out.path, STG1_PATH)

    def output(self):
        return luigi.LocalTarget(posify(STG1_PATH / f'{self.data_type}_after_luigi.csv'))


class DataFile(luigi.ExternalTask):
    data_type = luigi.Parameter(default='train')

    def output(self):
        return luigi.LocalTarget(posify(DATA_PATH / f'{self.data_type}_after_luigi.csv'))
