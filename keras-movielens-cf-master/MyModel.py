

import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense,BatchNormalization
from keras.models import Sequential
from keras.layers import Merge


class MyModel(Sequential):

    def __init__(self, max_userid, max_movieid,max_genre_dim,max_age_dim, k_factors, **kwargs):


        User_S = Sequential()
        User_S.add(Embedding(max_userid, k_factors, input_length=1))
        User_S.add(Reshape((k_factors,)))
        Item_S = Sequential()
        Item_S.add(Embedding(max_movieid, k_factors, input_length=1))
        Item_S.add(Reshape((k_factors,)))

        Genre_s = Sequential()
        embedding_size = 5
        Genre_s.add(Dense(embedding_size, activation="relu", use_bias=False, input_dim=max_genre_dim))

        Age_s = Sequential()
        Age_s.add(Embedding(max_age_dim, 3, input_length=1))
        Age_s.add(Reshape((3,)))

        Coor_s = Sequential()
        Coor_s.add(Dense(2, activation="relu", input_dim=2))


        p_dropout = 0.1

        super(MyModel, self).__init__(**kwargs)

        self.add(Merge([User_S, Item_S, Genre_s, Age_s, Coor_s], mode='concat'))
        self.add(Dropout(p_dropout))
        self.add(Dense(k_factors, activation='relu'))
        self.add(Dropout(p_dropout))
        # self.add(BatchNormalization())
        self.add(Dense(1, activation='linear'))


    def rate(self, Users, Movies,Genres_vectors,Ages,Coordinates):
        return self.predict([Users, Movies,Genres_vectors,Ages,Coordinates])[0][0]
