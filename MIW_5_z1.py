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
from tensorflow.keras.layers import Dense

# dane do przeliczenia
data = np.array([[0,0,0,0,0],
                 [1,0,0,1,0], 
                 [0,1,0,1,0], 
                 [1,1,0,0,1], 
                 [0,0,1,1,0], 
                 [1,0,1,0,1], 
                 [0,1,1,0,1], 
                 [1,1,1,1,1]])

train = np.zeros((8,3))
resArray = np.zeros((8,2))
descrData = np.array([["00"]
                    ,["10"]
                    ,["10"]
                    ,["01"]
                    ,["10"]
                    ,["01"]
                    ,["01"]
                    ,["11"]])

for i in range(8):
    train.itemset((i,0), data[i][0])
    train.itemset((i,1), data[i][1])  
    train.itemset((i,2), data[i][2])
    resArray.itemset((i,0), data[i][3])
    resArray.itemset((i,1), data[i][4])
print(descrData)

#przypisanie typu danych
data=data.astype("float64")
#określenie unikalnych wartości opisowych
descrData= LabelEncoder().fit_transform(descrData)

#podział na dane treningowe i testowe 1/3
data_train, data_test, descrData_train, descrData_test = train_test_split(data,descrData, test_size=0.33)

train_shape = data_train.shape[1]

#print(data)
#print(descData)

model = Sequential()
# print(model) importy działają

# warstwa wejściowa, (wielkość warstwy, funkcja aktywacji "relu", na końcu "softmax", rozkład normalny )
model.add(Dense(3, activation='relu', kernel_initializer='he_normal', input_shape=(train_shape,)))

# warstwa ukryta
model.add(Dense(2, activation='relu', kernel_initializer='he_normal'))

# warstwa wyjściowa (softmax wynik prawdopodobieństa, że obiekt jest klasyfikowany w danej klasie (1-3))
model.add(Dense(1, activation='softmax'))

# przygotowanie modelu do przeliczenia, algorytm optymalizujący - adam, fukncja crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# sparse_categorical_crossentropy
# categorical_crossentropy


# wrzucenie danych treningowych, 150 cykli
history = History()
model.fit(data_train, descrData_train, epochs=10000, batch_size=32, verbose=0,callbacks=[history],shuffle=True)
#print(history.history)
# sprawdzenie na zbiorze testowym
loss, acc = model.evaluate(data_test, descrData_test, verbose=0)


# wyniki
print("dokładność: ","\n",round(acc,4))
print("błędy","\n",round(loss,4))
#print(model.summary())


# wagi warstwy wejściowej
# print("warstwa wejściowa:","\n",model.layers[0].get_weights()[0],6)

# wagi warstwy ukrytej
wU = model.layers[1].get_weights()[0]
print("warstwa ukryta:","\n",wU)

# wagi warstwy wyjściowej
wW = model.layers[2].get_weights()[0]
print("warstwa wyjścia:","\n",wW)

# tablica błędów
lossGraph = history.history.get('loss')
print("tablica błędów:")
pp.plot(lossGraph)
pp.show()

print(acc)
print(model.summary())
#row = [1,1,0]
#yhat = model.predict([row])
#print(yhat)