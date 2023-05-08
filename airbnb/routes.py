import numpy as np
import json
from flask import render_template
from airbnb import app
from .models import Data


@app.route('/')
def index():
    mapbox_access_token = 'pk.eyJ1IjoiYm9yYXR1dHVtbHVlciIsImEiOiJjbGgwaGNwbmYwdG1xM2RqdTljbzdpZnk2In0.i0yKymv8S0CYFLZRytrHzw'

    with open('neighbourhoods.geojson', 'r') as f:
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
    # latlong = []
    # for neighbourhood, lat, lon in zip(all_data["neighbourhood_cleansed"], all_data["longitude"], all_data["latitude"]):
    #     latlong.append([lat, lon])

    neighlatlong = []
    for neighbourhood, lon, lat, seg in zip(all_data["neighbourhood_cleansed"], all_data["longitude"],
                                            all_data["latitude"], all_data["segment"]):
        neighlatlong.append([[lon, lat], neighbourhood, seg])

    # STATISTICS
    accommodates_price = all_data.groupby("accommodates")["price"].mean().reset_index()
    accommodates_list = accommodates_price["accommodates"].to_list()
    price_list = accommodates_price["price"].to_list()

    return render_template('index.html', mapbox_access_token=mapbox_access_token, points=neighlatlong,
                           accommodates=accommodates_list, price=price_list, neighbourhoods=neighbourhoods_geojson,
                           neighbourhoods_labels=neighbourhoods_labels, neighbourhoods_bbox=neighbourhoods_bbox)

