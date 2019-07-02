from pathlib import Path

HOME_PATH = Path.home()

ELO_PATH = HOME_PATH / 'Santander-Customer-Transaction-Prediction'
DATA_PATH = ELO_PATH / 'data'
PIPE_PATH = DATA_PATH / 'pipeline'
STG1_PATH = PIPE_PATH / 'stage_1'
STG2_PATH = PIPE_PATH / 'stage_2'
RESULT_PATH = PIPE_PATH / 'result'

