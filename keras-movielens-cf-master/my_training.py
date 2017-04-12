
# coding: utf-8

# In[151]:

import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np
import sys
sys.path.append("H:/WareHouse/MovieBigData/keras-movielens-cf-master/")
from CFModel import CFModel,BinaryEmbedding
from MyModel import MyModel
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from sklearn import preprocessing



def conv(val):
    if not val:
        return 0
    try:
        return np.float64(val)
    except:
        return np.float64(0)

def convGender(gender):
        if not gender:
            return 0

        if gender == "M":
            return 1
        else:
            return 0
# In[152]:

RATINGS_CSV_FILE = 'H:/WareHouse/MovieBigData/data/new_training_data.csv'
MODEL_WEIGHTS_FILE = 'my_weights.h5'
RNG_SEED = 1446557
k_factors=180


# In[155]:

ratings = pd.read_csv(RATINGS_CSV_FILE, 
                      sep=',', 
                      encoding='latin-1', 
                      usecols=["user_id","movie_id","rating","id","gender","age","occupation","zip_code","genre_name","genre_vector","year","coordinates"],converters={"year": conv,"gender":convGender})
max_userid = ratings['user_id'].drop_duplicates().max()
max_movieid = ratings['movie_id'].drop_duplicates().max()
max_year=ratings['year'].drop_duplicates().max()
max_occupation=ratings['occupation'].drop_duplicates().max()


max_features = 18

max_age = ratings['age'].drop_duplicates().max()



print(len(ratings), 'ratings loaded.')


# ## Creating Training Set
# 

# In[156]:
min_max_scaler = preprocessing.MinMaxScaler()

shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)
Users = shuffled_ratings['user_id'].values
print ('Users:', Users, ', shape =', Users.shape)

Movies = shuffled_ratings['movie_id'].values
print( 'Movies:', Movies, ', shape =', Movies.shape)
Ratings = shuffled_ratings['rating'].values
print ('Ratings:', Ratings, ', shape =', Ratings.shape)
min_max_scaler.fit_transform(Ratings)

Genders= shuffled_ratings['gender'].values
print ('Genders:', Genders, ', shape =', Genders.shape)

Ages= shuffled_ratings['age'].values
print ('Ages:', Ages, ', shape =', Ages.shape)
min_max_scaler.fit_transform(Ages)

Occupations= shuffled_ratings['occupation'].values
print ('Occupations:', Occupations, ', shape =', Occupations.shape)

Genres_vectors= shuffled_ratings['genre_vector'].str.split("|",expand=True).values
print ('Genres_vectors:', Genres_vectors, ', shape =', Genres_vectors.shape)

# Genres= shuffled_ratings['genre_name'].str.split("|").va
# print ('Genres:', Genres, ', shape =', Genres.shape)


Years= shuffled_ratings['year'].values
print ('Years:', Years, ', shape =', Years.shape)


Coordinates=shuffled_ratings['coordinates'].str.split(", ",expand=True).values
# print ('Coordinates:', Coordinates, ', shape =',Coordinates .shape)

Coordinates = min_max_scaler.fit_transform(Coordinates)


print ('Coordinates:', Coordinates, ', shape =',Coordinates.shape)


# In[158]:




# 

# In[164]:

model2=MyModel(max_userid, max_movieid,max_features,max_age,k_factors,max_occupation,max_year)
model2.compile(loss='mse', optimizer='adamax')


# In[165]:

callbacks = [EarlyStopping('val_loss', patience=3),
             ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
history = model2.fit([Users, Movies,Genres_vectors,Ages,Coordinates,Occupations,Years,Genders], Ratings, nb_epoch=30, validation_split=.25, verbose=2, callbacks=callbacks)


# In[139]:

loss = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                     'training': [ math.sqrt(loss) for loss in history.history['loss'] ],
                     'validation': [ math.sqrt(loss) for loss in history.history['val_loss'] ]})
ax = loss.ix[:,:].plot(x='epoch', figsize={7,10}, grid=True)
ax.set_ylabel("root mean squared error")
ax.set_ylim([0.0,3.0])


# In[141]:

min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))
print ('Minimum MAE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.fabs(min_val_loss)))


# In[ ]:

