from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from numpy import genfromtxt
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import preprocessingData

data = np.loadtxt('data.csv', delimiter= ",")
data, labels = preprocessingData(data)

x_train, X, y_train, y_test = train_test_split(data, labels,
                                                    test_size = 0.15,
                                                    stratify = labels,
                                                    random_state = 42,
                                                    shuffle = True)

train_data, C,  y_train, y_c = train_test_split(x_train, y_train,
                                                    test_size = 0.15,
                                                    stratify =y_train,
                                                    random_state = 42,
                                                    shuffle = True)
FeedForwardNetwork = Sequential()
FeedForwardNetwork.add(Dense(8, input_dim = len(train_data[0,:]), activation = 'relu'))
FeedForwardNetwork.add(Dense(4, activation='relu'))
FeedForwardNetwork.add(Dense(1, activation= 'sigmoid'))
FeedForwardNetwork.compile(loss= 'binary_crossentropy', optimizer='SGD', metrics = ['accuracy'] )
Ca = ModelCheckpoint(filepath = 'model.h5', monitor = 'val_loss', save_best_only = True, save_weights_only=True)
Cb = EarlyStopping(monitor= 'val_loss', mode = 'min', patience= 20)
history = model.fit(train_data, y_train, validation_data = (C, y_c), epochs= 200, batch_size = 10, callbacks=[Ca, Cb])