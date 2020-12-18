import math
import random as rd
from secrets import randbelow 
import numpy as np
import matplotlib.pyplot as pp

# funkcja aktywacji
def function(x):
    return 1 / (1 + math.e**(-x))

def neuronArray(x):
    if x==1: return np.array([1,0,0])
    if x==2: return np.array([0,1,0])
    if x==3: return np.array([0,0,1])    

# dane do przeliczenia
file = np.genfromtxt("C:\MIW\iris.data",dtype=["f8","f8","f8","f8","S15"],delimiter=",")
np.random.shuffle(file)

arrData = np.zeros((150,5))
data = np.zeros((120,5))
testData = np.zeros((30,5))

for i in range(150):
    arrData.itemset((i,0), file[i][0])
    arrData.itemset((i,1), file[i][1])  
    arrData.itemset((i,2), file[i][2])
    arrData.itemset((i,3), file[i][3])
    if file[i][4]==b"Iris-setosa": arrData.itemset((i,4), 1)
    if file[i][4]==b"Iris-virginica": arrData.itemset((i,4), 2)
    if file[i][4]==b"Iris-versicolor": arrData.itemset((i,4), 3)

data = arrData[:120]
testData = arrData[120:]

b1 = 0.5
in_put = np.random.rand(1,4)
in_put = np.array(in_put, dtype=np.float64)
hidden = np.random.rand(4,4)
hidden = np.array(hidden, dtype=np.float64)
output = np.random.rand(3,4)
output = np.array(output, dtype=np.float64)
res = np.random.rand(1,3)
res = np.array(res, dtype=np.float64)
ErrorTab = []
dec = 0.0

# zbiór treningowy
for i in range(100000):
	# warstwa wejścia
    RandRow = randbelow(120) #rd.randint(0,7)
    in_put = data[RandRow,[0,1,2,3]]
    neuronArray(data[RandRow,[4]])
    
	# warstwa ukrytwa
    hidden_out = function(in_put.dot(hidden.transpose()).transpose() + dec)
    out_out = function(hidden_out.dot(output.transpose()).transpose() + dec)  
   
    # warstwa wyjścia
    diff_out = ((-1) * np.subtract(res, out_out)) * (out_out*(1-out_out)) 
    diffCalcOut = (diff_out * (np.vstack((hidden_out,hidden_out,hidden_out)).transpose())).transpose()
    diffCalcOut = (output-(b1 * diffCalcOut)) 
    
    # korekty
    diff_hid = (diff_out.dot(output)) * (hidden_out*(1-hidden_out))
    diffCalcHid = (diff_hid * (np.vstack((in_put,in_put,in_put,in_put)).transpose())).transpose()
    diffCalcHid = (hidden-(b1*diffCalcHid)) 
    
    output = diffCalcOut
    hidden = diffCalcHid
  
# zbiór testowy
print("Tabela wyników:")
for i in range(30):
    in_put = testData[i,[0,1,2,3]]
    RandRow = randbelow(30)
    neuronArray(testData[RandRow,[4]])
    
    hidden_out = function(in_put.dot(hidden.transpose()).transpose() + dec)
    out_out = function(hidden_out.dot(output.transpose()).transpose() + dec)  
    errors = np.sum(b1*(res - out_out)**2)
    ErrorTab.append(errors)
        
    print("wejście: ",in_put, "oczekiwana wartość:", np.round(res), ",wyjście:", np.round(out_out), "Błędy:",  np.round(errors,6))

print("\n","ukryta:",hidden,"\n")
print("wyjście:",output,"\n")

pp.plot(ErrorTab)
pp.show()