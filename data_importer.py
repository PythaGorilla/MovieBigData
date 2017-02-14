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

def get_coordinates(zipcode):
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

def train(X,Y):
    gclf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    return gclf.fit(X,Y)

def writeCoordinates():
    df = get_movie_data()
    seen_code={}
    cor_list=[]

    for i in df["zip_code"] :
        if i not in seen_code.keys():
            try:
                cor = get_coordinates(i)
                cor_list.append(cor)
                seen_code[i] =cor
                print "new:",i,",",cor
            except:
                print "no response"
                cor_list.append((0,0))
        else:
            cor =seen_code.get(i)
            cor_list.append(cor)
            print "seen:",i,",",cor

    df["coordinates"] =cor_list
    df.to_csv("data/new_training_data.csv")




if __name__ == '__main__':
    writeCoordinates()
    # df =get_movie_data()
    # for i in df["zip_code"]:
    #     try:
    #         print get_coordinates(i)
    #     except:
    #         pass
    #print [get_coordinates(int(i)) for i in df["zip_code"] ]
    #print df["coordinates"]
