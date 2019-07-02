import luigi

from santander_luigi.pipeline.luigi_pipeline import GetSubmit

if __name__ == '__main__':
    task_test = GetSubmit()
    luigi.build([task_test], local_scheduler=True)
