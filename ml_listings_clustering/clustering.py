import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sqlalchemy import create_engine
import geopy

import sys
sys.path.append('../airbnb/helpers')
from airbnb.helpers.data_prep import *
from airbnb.helpers.pandas_options import set_pandas_options

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

engine = create_engine('postgresql://postgres:bora00254613@localhost/airbnb')
conn = engine.connect()

df_ = pd.read_csv("ml_listings_clustering/datasets/listings.csv")
df = df_.copy()

columns = ["neighbourhood_cleansed", 'property_type', 'room_type', 'accommodates', 'bathrooms_text',
           'bedrooms', 'beds', 'amenities', 'minimum_nights', 'availability_30', 'availability_60', 'availability_90',
           'availability_365', 'number_of_reviews', 'reviews_per_month', 'last_review',
           'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location', 'review_scores_value', 'host_since',
           'host_response_time', 'host_response_rate', 'host_is_superhost', 'host_has_profile_pic',
           'host_identity_verified', 'calculated_host_listings_count', 'price', 'latitude', 'longitude', 'listing_url',
           'name']

df = df[columns]

########################################################################################################################
#                                       DATA PRE-PROCESSING
########################################################################################################################
df["price"] = df["price"].apply(lambda x: float(x.split('$')[1].split('.00')[0].replace(',', '')))
df["host_since"] = pd.to_datetime(df["host_since"])
df["last_review"] = pd.to_datetime(df["last_review"])
##############################################
#               MISSING VALUES
##############################################
na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
for col in na_columns:
    df.dropna(subset=col, inplace=True)
##############################################
#                   OUTLIERS
##############################################
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
# df[num_cols].describe().T
for col in num_cols:
    df = remove_outliers(df, col, q1=0.01, q3=0.99)

df.reset_index(inplace=True)

df["bathrooms_text"].apply(lambda x: x.split(" ") if type(x) != float else x)
df["bathrooms"] = df["bathrooms_text"].apply(
    lambda x: re.findall(r'[\d\.\d]+', x)[0] if type(x) != float and re.findall(r'[\d\.\d]+', x) != [] else x)
df.loc[(df["bathrooms_text"].str.contains("Half") & (~df["bathrooms_text"].isnull())), "bathrooms"] = 1
df.loc[(df["bathrooms_text"].str.contains("half") & (~df["bathrooms_text"].isnull())), "bathrooms"] = 1
df["bathrooms"] = df["bathrooms"].astype(float)
df["bathrooms_text"] = df["bathrooms_text"].apply(
    lambda x: ''.join(filter(lambda y: not y.isdigit(), str(x))).lstrip().lstrip(".").lstrip())
df.loc[(df["bathrooms_text"] == "baths") | (df["bathrooms_text"] == "Half-bath"), "bathrooms_text"] = "bath"
df.loc[(df["bathrooms_text"] == "shared baths") | (
        df["bathrooms_text"] == "Shared half-bath"), "bathrooms_text"] = "shared bath"

df.loc[(df["bathrooms_text"] == "bath"), "bathrooms_text"] = "normal"
df.loc[(df["bathrooms_text"] == "shared bath"), "bathrooms_text"] = "shared"
df.loc[(df["bathrooms_text"] == "private bath"), "bathrooms_text"] = "private"

amenities_count_list = []
for i in range(len(df["amenities"])):
    count = len(df.loc[i, "amenities"].replace('["', "").replace('"]', "").replace('"', "").split(", "))
    amenities_count_list.append(count)
df["amenities"] = amenities_count_list

today_date = pd.to_datetime(pd.to_datetime("today").date())
df["host_since"] = df["host_since"].apply(lambda x: (today_date - x).days)
df["review_age"] = df["last_review"].apply(lambda x: (today_date - x).days)

########################################################################################################################
#                                       CLUSTERING DATA AND SENDING DATABASE
########################################################################################################################
cluster_columns = ["neighbourhood_cleansed", 'room_type', 'bedrooms', 'beds', 'bathrooms', 'bathrooms_text']
except_cluster_columns = [col for col in df.columns if col not in cluster_columns]
cluster_df = df[cluster_columns]
cluster_df_coor = df[["latitude", "longitude"]]
neighbourhood = cluster_df["neighbourhood_cleansed"].unique()
for i, v in enumerate(neighbourhood):
    cluster_df_tmp = cluster_df.copy()
    print(f"############ {v} #############")
    df_neighbourhood_tmp = cluster_df_tmp[cluster_df_tmp["neighbourhood_cleansed"] == v]
    cluster_df_tmp_last = pd.merge(df_neighbourhood_tmp, df[except_cluster_columns], left_index=True, right_index=True)
    columns = [col for col in df_neighbourhood_tmp if col != "neighbourhood_cleansed"]
    df_neighbourhood_tmp = df_neighbourhood_tmp[columns]
    num_cols = [col for col in df_neighbourhood_tmp if df_neighbourhood_tmp[col].dtypes != "O"]
    ms = MinMaxScaler()
    for col in num_cols:
        df_neighbourhood_tmp[col] = ms.fit_transform(df_neighbourhood_tmp[[col]])
    df_neighbourhood_tmp = one_hot_encoder(df_neighbourhood_tmp, ["room_type", "bathrooms_text"])

    if v == "Sultanbeyli":
        kmeans = KMeans(n_clusters=2).fit(df_neighbourhood_tmp)
        cluster_df_tmp_last["segment"] = kmeans.labels_ + 1
    elif v == "Esenler":
        kmeans = KMeans(n_clusters=3).fit(df_neighbourhood_tmp)
        cluster_df_tmp_last["segment"] = kmeans.labels_ + 1
    else:
        kmeans_tmp = KMeans().fit(df_neighbourhood_tmp)
        elbow = KElbowVisualizer(kmeans_tmp, k=(2, 20))
        elbow.fit(df_neighbourhood_tmp)
        print(elbow.elbow_value_)
        kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_neighbourhood_tmp)
        cluster_df_tmp_last["segment"] = kmeans.labels_ + 1

    cluster_df_tmp_last.to_sql(f"{v.lower()}_cluster", conn, if_exists="replace")


