import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.client import device_lib
from sklearn.utils import shuffle
import matplotlib.pyplot as pp
from keras.callbacks import History 

from tensorflow.keras import Sequential
#from keras.layers.core import Dense
from tensorflow.keras.layers import Dense

file = shuffle(pd.read_csv("C:\MIW\iris.data"))
#np.random.shuffle(file)

#podział na część liczbową opisową
data = file.values[:,:-1]
descrData = file.values[:, -1]

#przypisanie typu danych
data=data.astype("float64")
#określenie unikalnych wartości opisowych
descrData = LabelEncoder().fit_transform(descrData)

#podział na dane treningowe i testowe 1/3
data_train, data_test, descrData_train, descrData_test = train_test_split(data,descrData, test_size=0.33)

train_shape = data_train.shape[1]

#print(data)
#print(descData)

model = Sequential()
# print(model) importy działają

# warstwa wejściowa, (wielkość warstwy, funkcja aktywacji "relu", na końcu "softmax", rozkład normalny )
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(train_shape,)))

# warstwa ukryta
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))

# warstwa wyjściowa (softmax wynik prawdopodobieństa, że obiekt jest klasyfikowany w danej klasie (1-3))
model.add(Dense(3, activation='softmax'))

# przygotowanie modelu do przeliczenia, algorytm optymalizujący - adam, fukncja crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# wrzucenie danych treningowych, 150 cykli
history = History()
model.fit(data_train, descrData_train, epochs=150, batch_size=32, verbose=0,callbacks=[history])
#print(history.history)
# sprawdzenie na zbiorze testowym
loss, acc = model.evaluate(data_test, descrData_test, verbose=0)


# wyniki
print("dokładność: ","\n",round(acc,4))
print("błędy","\n",round(loss,4))
#print(model.summary())


# wagi warstwy wejściowej
# print("warstwa ukryta","\n",model.layers[0].get_weights()[0],6)

# wagi warstwy ukrytej
wU = model.layers[1].get_weights()[0]
print("warstwa ukryta","\n",wU)

# wagi warstwy wyjściowej
wW = model.layers[2].get_weights()[0]
print("warstwa wyjścia","\n",wW)

# tablica błędów
lossGraph = history.history.get('loss')
print("tablica błędów:")
pp.plot(lossGraph)
pp.show()