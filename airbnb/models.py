import pandas as pd
from sqlalchemy import create_engine

class Data:
    def __init__(self):
        self.conn = create_engine('postgresql://postgres:bora00254613@localhost/airbnb')

    def get_data(self, table):
        df = pd.read_sql_query(f"SELECT * FROM {table};", self.conn)
        df.rename(columns={"level_0": "index"}, inplace=True)
        return df