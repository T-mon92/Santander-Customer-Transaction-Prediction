import luigi

from santander_luigi.pipeline.luigi_pipeline import ExtractReal

if __name__ == '__main__':
    task_test = ExtractReal()
    luigi.build([task_test], local_scheduler=True)
