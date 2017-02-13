import pandas as pd
import random, datetime
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import os
import subprocess
from geopy.geocoders import Nominatim
from sklearn.preprocessing import OneHotEncoder

def encode_one_hot(X):
    enc = OneHotEncoder()
    return enc.transform(X)

def get_cordinates(zipcode):
    geolocator = Nominatim()
    location = geolocator.geocode(zipcode)
    return (location.latitude, location.longitude)


def get_movie_data():
    if os.path.exists("data/training_data.csv"):
        print("-- movies.csv found locally")
        df = pd.read_csv("data/training_data.csv", index_col=0)

    return df

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)
