from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polyline

# DATETIME FEATURES

HOUR = 3600

city_id_to_delda: Dict[int, timedelta] = {
    338: timedelta(seconds=HOUR*0),
    22402: timedelta(seconds=HOUR*2),
    22406: timedelta(seconds=HOUR*2),
    22394: timedelta(seconds=HOUR*1),
    1078: timedelta(seconds=HOUR*0),
    22390: timedelta(seconds=HOUR*1),
    22430: timedelta(seconds=HOUR*1),
    22438: timedelta(seconds=HOUR*2),
}

holidays: List[Tuple[int, int]] = [
    (1, 1),
    (2, 1),
    (3, 1),
    (6, 1),
    (7, 1),
    (8, 1),
    (24, 2),
    (9, 3),
]


TEST_COLUMNS = ['Id', 'main_id_locality', 'ETA', 'OrderedDate', 'latitude',
       'del_latitude', 'longitude', 'del_longitude', 'EDA', 'center_latitude',
       'center_longitude', 'route', 'OrderedDay', 'OrderedHour',
       'OrderedMonth', 'IsHoliday', 'points',
       'min_from_center', 'max_from_center'
       ]

FEATURES = ["OrderedDay", "OrderedHour", "OrderedMonth", "IsHoliday", 
            "AbsLatitudeChange", "AbsLongitudeChange", "LatitudeFromCenter", "LongitudeFromCenter",
            "del_LatitudeFromCenter", "del_LongitudeFromCenter", "points",
             "min_from_center", "max_from_center"
             ]

ALL_COLUMNS = TEST_COLUMNS + FEATURES
    

def to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["OrderedDate"] = pd.to_datetime(df["OrderedDate"])
    return df


def get_local_time(df: pd.DataFrame) -> pd.DataFrame:
    for city, delta in city_id_to_delda.items():
        df.loc[df["main_id_locality"] == city, ["OrderedDate"]] += delta
    return df


def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["OrderedDay"] = df["OrderedDate"].apply(lambda x: x.dayofweek)
    df["OrderedHour"] = df["OrderedDate"].apply(lambda x: x.hour)
    df["OrderedMonth"] = df["OrderedDate"].apply(lambda x: x.month)
    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    def is_holiday(row: pd.Series):
        if row["OrderedDay"] in {5, 6}:
            return 1
        if (row["OrderedDate"].day, row["OrderedDate"].month) in holidays:
            return 1
        return 0

    df["IsHoliday"] = df.apply(is_holiday, axis=1)
    return df


def add_all_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    functions = (
        to_datetime,
        get_local_time,
        create_datetime_features,
        add_holidays
    )
    for f in functions:
        df = f(df)

    return df

# GEOSPATIAL

def add_abs_coordinate_change(df: pd.DataFrame) -> pd.DataFrame:
    df["AbsLatitudeChange"] = np.abs(df["del_latitude"] - df["latitude"])
    df["AbsLongitudeChange"] = np.abs(df["del_longitude"] - df["longitude"])
    return df


def normalize_to_center(df: pd.DataFrame) -> pd.DataFrame:
    df["LatitudeFromCenter"] = np.abs(df["latitude"] - df["center_latitude"])
    df["LongitudeFromCenter"] = np.abs(
        df["longitude"] - df["center_longitude"])
    df["del_LatitudeFromCenter"] = np.abs(
        df["del_latitude"] - df["center_latitude"])
    df["del_LongitudeFromCenter"] = np.abs(
        df["del_longitude"] - df["center_longitude"])
    return df


def decode_route(df: pd.DataFrame) -> pd.DataFrame:
    def get_route(poly: str):
        route = polyline.decode(poly, 5)
        route = np.array(route)
        return route

    df["route_decode"] = df["route"].apply(get_route)
    return df


def get_vertex_number(df: pd.DataFrame) -> pd.DataFrame:
    df["points"] = df["route_decode"].apply(lambda x: len(x))
    return df


def get_min_max_from_center(df: pd.DataFrame) -> pd.DataFrame:
    def min_from_center(row):
        try:
            route = row['route_decode']
            sx = row['latitude']
            sy = row['longitude']
            fx = row['del_latitude']
            fy = row['del_longitude']
            # center_latitude center_longitude
            cx = row['center_latitude']
            cy = row['center_longitude']

            diff_sqs = (route - np.array((cx, cy))) ** 2
            dists = np.sqrt(diff_sqs.sum(axis=1))
            return dists.min(), dists.max()
        except Exception as e:
            print(e)
            return (0, 0)

    df["min_from_center"], df["max_from_center"] = zip(*df.apply(min_from_center, axis=1))
    return df


def add_all_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    functions = (
        normalize_to_center,
        add_abs_coordinate_change,
        decode_route,
        get_vertex_number,
        get_min_max_from_center,
    )
    for f in functions:
        df = f(df)

    return df


def filter_columns(df: pd.DataFrame, is_test=False) -> pd.DataFrame:
    if is_test:
        return df[ALL_COLUMNS]
    return df[ALL_COLUMNS + ["RTA"]]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    df = df.fillna("")
    print("Add datetime features")
    df = add_all_datetime_features(df)
    print("Add geo features")
    df = add_all_geo_features(df)
    df = filter_columns(df, is_test=args.test)
    df = df.fillna(0)
    print("Saving")
    df.to_csv(args.output_file, index=None)
    print("Done", flush=True)
