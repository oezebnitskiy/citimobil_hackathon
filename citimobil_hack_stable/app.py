# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of showing geographic data."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

DATE_TIME = "date/time"
ERROR = 'error'

st.title("Citimobil Pickups and Destinations")
st.markdown(
"""
Visualisation of pickup/destination points in various cities, based on previous dataset.

Based on Uber visualisation of pickups in New York
""")

@st.cache(persit=True)
def load_data(nrows):
    cities = {'PermPickup':'perm.csv',
        'PermDestination':'perm_des.csv',
        'KazanPickup':'kazan.csv',
        'SamaraPickup':'samara.csv',
        'SaratovPickup':'saratov.csv',
        'KazanDestination':'kazan_des.csv',
        'SamaraDestination':'samara_des.csv',
        'SaratovDestination':'saratov_des.csv'}
    res = {}
    for city in cities:
        for pickup_des in ['Pickup', 'Destination']:
            filename = cities[city] 
            print(filename)
            data = pd.read_csv(filename, nrows=nrows)
            lowercase = lambda x: str(x).lower()
            data.rename(lowercase, axis="columns", inplace=True)
            print(data.head())
            data[DATE_TIME] = pd.to_datetime(data[DATE_TIME])
            res[city]=data
    return res


res = load_data(100000)

option = st.selectbox('City?',('Perm', 'Kazan', 'Samara', 'Saratov'))
pickup_des = st.selectbox('From/To?',('Pickup', 'Destination'))
data = res[option+pickup_des]

batch = st.slider("ETA Error", 0, 10)

data_map = data[data[ERROR] >= batch]

st.subheader("Error in ETA is more than N * 100 Seconds")
midpoint = (np.average(data_map["lat"]), np.average(data_map["lon"]))

st.write(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state={
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 11,
        "pitch": 50,
    },
    layers=[
        pdk.Layer(
            "HexagonLayer",
            data=data_map,
            get_position=["lon", "lat"],
            radius=100,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
    ],
))

st.subheader("Breakdown by error")
filtered = data

hist = np.histogram(data['error'], bins=10, range=(0, 10))[0]
chart_data = pd.DataFrame({"Batches": range(10), "times": hist})

st.altair_chart(alt.Chart(chart_data)
    .mark_area(
        interpolate='step-after',
    ).encode(
        x=alt.X("Batches:Q", scale=alt.Scale(nice=False)),
        y=alt.Y("times:Q"),
        tooltip=['Batches', 'times']
    ), use_container_width=True)

if st.checkbox("Show raw data", False):
    st.subheader("Raw data")
    st.write(data)
