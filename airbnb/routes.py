import numpy as np
import joblib
import pandas as pd
from flask import render_template, request, jsonify
from airbnb import app
from .models import Data
from ml_price_prediction.model_pipeline import data_preprocessing


@app.route('/')
def index():
    mapbox_access_token = 'pk.eyJ1IjoiYm9yYXR1dHVtbHVlciIsImEiOiJjbGgwaGNwbmYwdG1xM2RqdTljbzdpZnk2In0.i0yKymv8S0CYFLZRytrHzw'

    bathroom_types = pd.read_csv("airbnb/static/data/list_bathroom_types.csv")
    host_response_times = pd.read_csv("airbnb/static/data/list_host_response_times.csv")
    property_types = pd.read_csv("airbnb/static/data/list_property_types.csv")
    room_types = pd.read_csv("airbnb/static/data/list_room_types.csv")

    with open('airbnb/static/data/neighbourhoods.geojson', 'r') as f:
        neighbourhoods_geojson = f.read()

    data = Data()
    all_data = data.get_data(table="all_data")
    neighbourhoods = data.get_data(table="neighbourhoods")

    # NEIGHBOURHOODS
    neighbourhoods_bbox = {}
    for names, minx, miny, maxx, maxy in zip(neighbourhoods["neighbourhood"], neighbourhoods["minx"],
                                             neighbourhoods["miny"], neighbourhoods["maxx"], neighbourhoods["maxy"]):
        neighbourhoods_bbox[names] = [[minx, miny], [maxx, maxy]]

    neighbourhoods_labels = sorted(list(neighbourhoods_bbox.keys()))

    # POINTS
    point_properties = []
    for url, name, neighbourhood, lon, lat, seg in zip(all_data["listing_url"],
                                                       all_data["name"],
                                                       all_data["neighbourhood_cleansed"],
                                                       all_data["longitude"],
                                                       all_data["latitude"],
                                                       all_data["segment"]):
        point_properties.append([[lon, lat], neighbourhood, seg, url, name])

    # STATISTICS
    accommodates_price = all_data.groupby("accommodates")["price"].mean().reset_index()
    accommodates_list = accommodates_price["accommodates"].to_list()
    accommodates_price_list = accommodates_price["price"].to_list()

    avg_price_per_neigbourhood = all_data.groupby("neighbourhood_cleansed")["price"].mean().reset_index().sort_values(
        "price", ascending=False)
    avg_price_per_neigbourhood_neigbourhood = avg_price_per_neigbourhood["neighbourhood_cleansed"].to_list()
    avg_price_per_neigbourhood_price = avg_price_per_neigbourhood["price"].to_list()

    index = [i for i, v in enumerate(all_data["property_type"]) if
             v in ["Entire rental unit", "Private room in rental unit", "Entire condo", "Entire serviced apartment"]]
    property_type_statistic = all_data[all_data.index.isin(index)]["property_type"].value_counts().reset_index().rename(
        columns={"index": "property_type",
                 "property_type": "property_type_count"})
    property_type_statistic_types = property_type_statistic["property_type"].to_list()
    property_type_statistic_count = property_type_statistic["property_type_count"].to_list()

    return render_template('index.html',
                           mapbox_access_token=mapbox_access_token,
                           points=point_properties,
                           accommodates=accommodates_list,
                           accommodates_price_list=accommodates_price_list,
                           neighbourhoods=neighbourhoods_geojson,
                           neighbourhoods_labels=neighbourhoods_labels,
                           neighbourhoods_bbox=neighbourhoods_bbox,
                           bathroom_types=list(bathroom_types["Bathroom Type"]),
                           host_response_times=list(host_response_times["Host Response Time"]),
                           property_types=list(property_types["Property Types"]),
                           room_types=list(room_types["Room Types"]),
                           avg_price_per_neigbourhood_neigbourhood=avg_price_per_neigbourhood_neigbourhood,
                           avg_price_per_neigbourhood_price=avg_price_per_neigbourhood_price,
                           property_type_statistic_types=property_type_statistic_types,
                           property_type_statistic_count=property_type_statistic_count)


@app.route('/sendmapdata', methods=['GET'])
def predict():
    args = request.args

    neighbourhood_cleansed = args.get('neighbourhood_cleansed')
    property_type = args.get('property_type')
    room_type = args.get('room_type')
    host_response_time = args.get('host_response_time')
    bathrooms_text = args.get('bathrooms_text')
    amenities = 30
    accommodates = int(args.get('accommodates'))
    beds = int(args.get('beds'))
    bedrooms = int(args.get('bedrooms'))
    bathrooms = int(args.get('bathrooms'))
    minimum_nights = int(args.get('minimum_nights'))
    availability_30 = int(args.get('availability_30'))
    availability_365 = int(args.get('availability_365'))
    number_of_reviews = int(args.get('number_of_reviews'))
    review_age = 100
    reviews_per_month = 1
    review_scores_rating = float(args.get('review_scores_rating'))
    review_scores_accuracy = float(args.get('review_scores_accuracy'))
    review_scores_cleanliness = float(args.get('review_scores_cleanliness'))
    review_scores_checkin = float(args.get('review_scores_checkin'))
    review_scores_communication = float(args.get('review_scores_communication'))
    review_scores_location = float(args.get('review_scores_location'))
    review_scores_value = float(args.get('review_scores_value'))
    host_since = int(args.get('host_since'))
    host_is_superhost = 'f'
    host_identity_verified = 't'
    host_response_rate = args.get('host_response_rate')
    calculated_host_listings_count = int(args.get('calculated_host_listings_count'))
    longitude = float(args.get('coordx'))
    latitude = float(args.get('coordy'))

    values = [neighbourhood_cleansed, room_type, bedrooms, beds, bathrooms, bathrooms_text,
              property_type, accommodates, amenities, minimum_nights, availability_30,
              availability_365, number_of_reviews, reviews_per_month,
              review_scores_rating, review_scores_accuracy, review_scores_cleanliness,
              review_scores_checkin, review_scores_communication, review_scores_location,
              review_scores_value, host_since, host_response_time, host_response_rate,
              host_is_superhost, host_identity_verified, review_age,
              calculated_host_listings_count, latitude, longitude]

    data = Data()
    data.insert_data(table="data_to_predict", values=values)
    df = data.get_data(table="data_to_predict")
    model = joblib.load("lgbm.pkl")
    X, y = data_preprocessing(df)
    X_sample = X[-1:]
    prediction_price = round(np.expm1(model.predict(X_sample))[0])
    print(prediction_price)
    data.update_price(table="data_to_predict", price_predicted=prediction_price)

    return jsonify({'prediction_price': prediction_price})
