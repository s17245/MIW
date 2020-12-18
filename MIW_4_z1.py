import math
import random as rd
from secrets import randbelow 
import numpy as np
import matplotlib.pyplot as pp

#Dane do przeliczenia
data = np.array([[0,0,0,0,0],
                 [1,0,0,1,0], 
                 [0,1,0,1,0], 
                 [1,1,0,0,1], 
                 [0,0,1,1,0], 
                 [1,0,1,0,1], 
                 [0,1,1,0,1], 
                 [1,1,1,1,1]])
 
#funkcja aktywacji
def function(x):
    return 1 / (1 + math.e**(-x))

b1 = 0.5
in_put = np.random.rand(1,3)
in_put = np.array(in_put, dtype=np.float64)
hidden = np.random.rand(3,3)
hidden = np.array(hidden, dtype=np.float64)
output = np.random.rand(2,3)
output = np.array(output, dtype=np.float64)
res = np.random.rand(1,2)
res = np.array(res, dtype=np.float64)
ErrorTab = []
dec = 0.0

#Obliczanie błędów
for i in range(100000000):
	# warstwa wejścia
    RandRow = randbelow(8) #rd.randint(0,7)
    in_put = data[RandRow,[0,1,2]]
    res = data[RandRow,[3,4]] 
    
	# warstwa ukrytwa
    hidden_out = function(in_put.dot(hidden.transpose()).transpose() + dec)
    out_out = function(hidden_out.dot(output.transpose()).transpose() + dec)  
    errors = np.sum(b1*(res - out_out)**2)
    ErrorTab.append(errors)
    
    # warstwa wyjścia
    diff_out = ((-1) * np.subtract(res, out_out)) * (out_out*(1-out_out)) 
    diffCalcOut = (diff_out * (np.vstack((hidden_out,hidden_out)).transpose())).transpose()
    diffCalcOut = (output-(b1 * diffCalcOut)) 
    
    # korekty
    diff_hid = (diff_out.dot(output)) * (hidden_out*(1-hidden_out))
    diffCalcHid = (diff_hid * (np.vstack((in_put,in_put,in_put)).transpose())).transpose()
    diffCalcHid = (hidden-(b1*diffCalcHid)) 
    
    output = diffCalcOut
    hidden = diffCalcHid
  
# tabela wyników
print("Tabela wyników:")
for i in range(7):
    in_put = data[i,[0,1,2]]
    res = data[i,[3,4]] 
    
    #Obliczenie odpowiedzi sieci
    hidden_out = function(in_put.dot(hidden.transpose()).transpose() + dec)
    out_out = function(hidden_out.dot(output.transpose()).transpose() + dec)  
    print("wejście: ",in_put, "oczekiwana wartość:", res, ",wyjście:", np.round(out_out), "Błędy:",  np.round(errors,6))

print("\n","ukryta:",hidden,"\n")
print("wyjście:",output,"\n")

pp.plot(ErrorTab)
pp.show()