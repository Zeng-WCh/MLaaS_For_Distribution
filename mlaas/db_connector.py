import logging
import pymysql as sql
import sys

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any
from utils import init_logger, reset_logger_level

logger = init_logger('DB Connector', 'db_connector.log', sys.stdout)


@dataclass
class Dataline(object):
    model_choice: str | None = None
    training_status: str = "pending"
    created_by: int | None = None
    completed_at: None | Any = None
    model_location: None | str = None
    model_name: None | str = None

    @staticmethod
    def line_description():
        return ', '.join([k for k in Dataline.__dataclass_fields__])

    @staticmethod
    def create_placeholder():
        return ', '.join(['%s' for _ in Dataline.__dataclass_fields__])

    def to_sql_arguments(self):
        # return tuple([getattr(self, k) if getattr(self, k) is not None else 'null' for k in Dataline.__dataclass_fields__])
        return [getattr(self, k) for k in Dataline.__dataclass_fields__]


class DatabaseConnector(object):
    def __init__(self, **kwargs):
        try:
            self.host = kwargs['host']
            self.port = kwargs['port']
            self.user = kwargs['user']
            self.password = kwargs['password']
            self.database = kwargs['database']
        except KeyError as e:
            logger.error(f'KeyError when initializing db: {e}')
            self.connection = None
        else:
            self.connection = self.__connect()

    def reset_connection(self, host: str | None, port: int | None, user: str | None, password: str | None, database: str | None):
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port
        if user is not None:
            self.user = user
        if password is not None:
            self.password = password
        if database is not None:
            self.database = database

        if self.connection is not None:
            self.connection.close()
        self.connection = self.__connect()

    def __connect(self):
        logger.info(f'Connecting to database({self.host}:{self.port})...')
        try:
            connection = sql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
        except Exception as e:
            logger.error(f'Try to connect to database failed: {e}')
            connection = None
            raise e
        logger.info(f'Connected to database({self.host}:{self.port})')
        return connection

    def close_db(self):
        if self.connection is None:
            logger.warn(f'No db is connected')
        else:
            logger.info('Before close db connect, trying to commit...')
            self.commit()
            logger.info(
                f'Closing database connection({self.host}:{self.port})')
            self.connection.close()
            self.connection = None

    def is_connected(self):
        return self.connection is not None

    def commit(self):
        logger.info('Committing...')
        try:
            self.connection.commit()
        except Exception as e:
            logger.error(f'Commit failed: {e}')
            logger.error(f'Rolling back...')
            self.connection.rollback()
            return False
        return True

    '''
    execute the query and commit the result
    '''

    def execute_query(self, format_sql: str, required_cursor: bool = False, *args):
        if not self.is_connected():
            logger.error('No db is connected')
            return False, None
        logger.info(f'Executing query: {format_sql} with args: {args}')
        cursor = self.connection.cursor()

        cursor.execute(format_sql, args)
        result = self.commit()
        if not required_cursor:
            cursor.close()
            return result, None

        return result, cursor


if __name__ == '__main__':
    parser = ArgumentParser('MySQL Database Connector')

    parser.add_argument('--host', type=str, required=True,
                        help='Database Host')
    parser.add_argument('--port', type=int, default=3306, help='Database Port')
    parser.add_argument('--user', type=str, required=True,
                        help='Database User')
    parser.add_argument('--password', type=str,
                        required=True, help='Database Password')
    parser.add_argument('--database', type=str,
                        required=True, help='Database Name')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    if args.debug:
        reset_logger_level(logger, logging.DEBUG)

    # db = get_db_connection(args.host, args.port, args.user, args.password, args.database)
    db = DatabaseConnector(host=args.host, port=args.port, user=args.user,
                           password=args.password, database=args.database)
    username = ''
    # Test

    command = f'SELECT username FROM user WHERE id = 1;'
    status, cursor = db.execute_query(command, True)
    if status:
        logger.info(f'Get record successfully')
        if cursor is not None:
            for row in cursor.fetchall():
                logger.info(f'Get record: {row}')
                username = row[0]
            cursor.close()
    else:
        logger.error(f'Get record failed')

    line = Dataline('SVM', False, username)

    command = f"INSERT INTO training_records (submit_date, {line.line_description()}) VALUES (NOW(), {line.create_placeholder()});"
    status, cursor = db.execute_query(command, True, *line.to_sql_arguments())
    if status:
        id = cursor.lastrowid
        cursor.close()
        logger.info(f'Insert record successfully')
        logger.debug(f'Last row id: {id}')
    else:
        logger.error(f'Insert record failed')
    if cursor is not None:
        cursor.close()

    command = f'SELECT username FROM user WHERE id = 1;'
    status, cursor = db.execute_query(command, True)
    if status:
        logger.info(f'Get record successfully')
        for row in cursor.fetchall():
            logger.info(f'Get record: {row}')
    else:
        logger.error(f'Get record failed')

    db.close_db()
