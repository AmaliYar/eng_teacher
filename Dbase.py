from abc import ABC
import pg8000
DEFAULT_QDRANT_NAME = 'embeds'
DEFAULT_POSTGRES_CONNECTION_CONFIG = {
    'database': 'agent',
    'user': 'postgres',
    'password': '123',
    'host': 'localhost',
    'port': 5432
            }


class Database(ABC):

    def send_query(self, query, params):
        pass

    def end_session(self):
        pass

    def transform_to_retriever(self):
        pass


class Postgres(Database):
    config: dict
    connection: pg8000.Connection
    cursor: pg8000.Cursor

    def __init__(self, params=None):
        if not params:
            params = DEFAULT_POSTGRES_CONNECTION_CONFIG
        try:
            self.connection = pg8000.connect(**params)
            self.cursor = self.connection.cursor()
            print('connected to db successful')
        except pg8000.Error as err:
            print(f"An error occurred: {err}")

    def send_query(self, query: str, params) -> tuple:
        try:
            return self.cursor.execute(query, params).fetchall()
        except pg8000.Error as err:
            print(f"An error occurred: {err}")
            return tuple()

    def end_session(self):
        self.cursor.close()
        self.connection.close()

