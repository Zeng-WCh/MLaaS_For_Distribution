# Description: Configuration file for the backend

TRAINER_LOG_INTERVAL = 5  # How frequently to log information
TERMINATE_AT_ITER = 300  # for early stopping when debugging
PS_AVERAGE_EVERY_N = 25  # How often to average models between trainers

BROADCAST_PORT = 45950  # Port for broadcasting
CONNECTION_TIMEOUT = 30  # How long to wait for a connection to the worker

# EXTRA_EPOCHS = 5  # How many extra epochs to train after all other workers have finished

DB_NAME = 'mlaas'
SQL_INSERT_COMMAND = r"insert into "

DB_CONFIG = {
    'host': '192.168.43.114',
    'port': 3306,
    'user': 'user',
    'password': 'password',
    'database': 'mlaas',
    'table': 'training_records',
}

MODEL_DIRECTORY = './exported_models'
