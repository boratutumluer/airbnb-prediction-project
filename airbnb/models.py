import pandas as pd
from sqlalchemy import create_engine


class Data:
    def __init__(self):
        self.conn = create_engine('postgresql://postgres:bora00254613@localhost/airbnb')

    def get_data(self, table):
        df = pd.read_sql_query(f"SELECT * FROM {table};", self.conn)
        df.rename(columns={"level_0": "index"}, inplace=True)
        return df

    def insert_data(self, table, values):
        columns = ["neighbourhood_cleansed", "room_type", "bedrooms", "beds", "bathrooms", "bathrooms_text",
                   "property_type", "accommodates", "amenities", "minimum_nights", "availability_30",
                   "availability_365", "number_of_reviews", "reviews_per_month",
                   "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
                   "review_scores_checkin", "review_scores_communication", "review_scores_location",
                   "review_scores_value", "host_since", "host_response_time", "host_response_rate",
                   "host_is_superhost", "host_identity_verified", "review_age",
                   "calculated_host_listings_count", "latitude", "longitude"]
        # price = 1000
        # values = values + [price]
        text = ''
        for i in values:
            if type(i) == str:
                text += "'" + i + "',"
            else:
                text += str(i) + ","
        text = text.rstrip(',')

        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({text})"
        self.conn.execute(query)
        print("INSERT DONE!")

    def update_price(self, table, price_predicted):
        id = pd.read_sql_query("SELECT max(id) FROM data_to_predict;", con=self.conn).values[0][0]
        self.conn.execute(f"UPDATE {table} SET price = {price_predicted} where id = {id}")
