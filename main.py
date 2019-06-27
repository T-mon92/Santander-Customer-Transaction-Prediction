import luigi

from santander_luigi.data.load_data import ExtractReal

if __name__ == '__main__':
    task_test = ExtractReal('test')
    luigi.build([task_test], local_scheduler=True)
