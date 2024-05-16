import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
df = pd.read_csv('data.csv')

#Xs = df['X'].to_numpy(dtype=float)
#Ys = df['Y'].to_numpy(dtype=float)
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Y']), df['Y'], test_size=0.2, random_state=42)
#print(Xs.shape)
Xs_train = X_train.to_numpy()
Ys_train = y_train.to_numpy()
Xs_test = X_test.to_numpy()
Ys_test = y_test.to_numpy()

#print(X_test)

#Definition de notre couche
Layer01 = Dense(units=1, input_shape=[1])

#Definition du modele
model = Sequential([Layer01])

#Definition de la loss function et l'optimizer (Stochastic Gradient Descent)
model.compile(optimizer='sgd', loss='mean_squared_error')

#Entrainement du modele
model.fit(Xs_test, Ys_test, epochs=100)

#Recuperations des paramètres après entrainement
print("Voici ce qui à été appris :{}".format(Layer01.get_weights()))

#evaluation du model
model.evaluate(Xs_test, Ys_test, verbose=2)
