import luigi

from santander_luigi.data.load_data import LoadData

if __name__ == '__main__':
    task_train = LoadData('train')
    task_test = LoadData('test')
    luigi.build([task_train, task_test], local_scheduler=True)
