import numpy as np
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
import sys
sys.path.append("H:/WareHouse/MovieBigData/keras-movielens-cf-master/")
from CFModel import CFModel,BinaryEmbedding

model = Sequential()
#model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.
model.add(Dense(12,activation="relu",use_bias=False,input_dim=10))
print(model.output_shape)
model.add(Reshape((3,4)))

input_array = np.random.randint(1000, size=(4, 10))

model.compile('rmsprop', 'mse')

print(model.output_shape)

output_array = model.predict(input_array)
print(output_array)