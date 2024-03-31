SQL_CONFIG = {
    'host': '192.168.43.114',
    'port': 3306,
    'user': 'user',
    'password': 'password',
    'db': 'mlaas'
}

SQL_URI = f'mysql+pymysql://{SQL_CONFIG["user"]}:{SQL_CONFIG["password"]}@{SQL_CONFIG["host"]}:{SQL_CONFIG["port"]}/{SQL_CONFIG["db"]}'

SECRET_KEY = 'your_secret_key'

API_SERVER = 'http://localhost:8000'
