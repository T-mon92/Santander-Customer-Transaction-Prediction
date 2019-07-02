import luigi

from santander_luigi.pipeline.luigi_pipeline import GetSubmit

if __name__ == '__main__':
    task_santander = GetSubmit()
    luigi.build([task_santander], local_scheduler=True)
