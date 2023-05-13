import numpy as np
import json
import pandas as pd
from flask import render_template, request
from airbnb import app
from .models import Data


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
    price_list = accommodates_price["price"].to_list()

    # if request.method == 'POST':
    #
    #     # Dropdowns
    #     neighbourhood = request.form.get('district')
    #     propertytype = request.form.get('property_types')
    #     if propertytype == "":
    #         propertytype = "House"
    #     roomtype = request.form.get('room_types')
    #     if roomtype == "":
    #         roomtype = "Entire home/apt"
    #     hostresponsetime = request.form.get('response_time')
    #     if hostresponsetime == "":
    #         hostresponsetime = "within a day"
    #     bathroomtype = request.form.get('bathroom_types')
    #     if bathroomtype == "":
    #         bathroomtype = "normal"
    #
    #     # Text input
    #     accommodates = request.form['accommodates']
    #     if accommodates == "":
    #         accommodates = 2
    #     bedrooms = request.form['bedrooms']
    #     if bedrooms == "":
    #         bedrooms = 1
    #     beds = request.form['beds']
    #     if beds == "":
    #         beds = 1
    #     minimum_night = request.form['minimum_night']
    #     if minimum_night == "":
    #         minimum_night = 1
    #     availability_30 = request.form['availability_30']
    #     if availability_30 == "":
    #         availability_30 = 15
    #     availability_60 = 12
    #     availability_90 = 12
    #     availability_365 = request.form['availability_365']
    #     if availability_365 == "":
    #         availability_365 = 150
    #     number_of_score = request.form['number_of_score']
    #     if number_of_score == "":
    #         number_of_score = 50
    #     reviews_per_month = 0.22
    #     reviewscore = request.form['reviewscore']
    #     if reviewscore == "":
    #         reviewscore = 80
    #     accuracyscore = request.form['accuracyscore']
    #     if accuracyscore == "":
    #         accuracyscore = 4
    #     cleanliness = request.form['cleanliness']
    #     if cleanliness == "":
    #         cleanliness = 4
    #     checkinscore = request.form['checkinscore']
    #     if checkinscore == "":
    #         checkinscore = 4
    #     interactionscore = request.form['interactionscore']
    #     if interactionscore == "":
    #         interactionscore = 4
    #     locationscore = request.form['locationscore']
    #     if locationscore == "":
    #         locationscore = 4
    #     valuescore = request.form['valuescore']
    #     if valuescore == "":
    #         valuescore = 4
    #     host_since = request.form['host_since']
    #     if host_since == "":
    #         host_since = 100
    #     responserate = request.form['responserate']
    #     if responserate == "":
    #         responserate = 90
    #     nlisting = request.form['nlisting']
    #     if nlisting == "":
    #         nlisting = 1
    #
    #     df_input = pd.DataFrame([[neighbourhood, propertytype, roomtype,
    #                               hostresponsetime, accommodates, num_bedrooms, num_beds,
    #                               min_nights, availability_30, availability_60, availability_90, availability_365,
    #                               num_reviews, reviews_per_month, review_scores_rating, review_scores_accuracy,
    #                               review_scores_cleanliness, review_scores_checkin, review_scores_communication,
    #                               review_scores_location, review_scores_value, host_response_rate,
    #                               ]],
    #                             columns=['Country', 'City', 'Neighbourhood Cleansed', 'Property Type',
    #                                      'Room Type', 'Bed Type', 'Cancellation Policy', 'Host Response Time',
    #                                      'Accommodates', 'Bedrooms', 'Beds', 'Minimum Nights', 'Availability 30',
    #                                      'Availability 60', 'Availability 90', 'Availability 365',
    #                                      'Number of Reviews', 'Reviews per Month', 'Review Scores Rating',
    #                                      'Review Scores Accuracy', 'Review Scores Cleanliness',
    #                                      'Review Scores Checkin', 'Review Scores Communication',
    #                                      'Review Scores Location', 'Review Scores Value', 'Host Response Rate'])


    return render_template('index.html',
                           mapbox_access_token=mapbox_access_token,
                           points=point_properties,
                           accommodates=accommodates_list,
                           price=price_list,
                           neighbourhoods=neighbourhoods_geojson,
                           neighbourhoods_labels=neighbourhoods_labels,
                           neighbourhoods_bbox=neighbourhoods_bbox,
                           bathroom_types=list(bathroom_types["Bathroom Type"]),
                           host_response_times=list(host_response_times["Host Response Time"]),
                           property_types=list(property_types["Property Types"]),
                           room_types=list(room_types["Room Types"]))

