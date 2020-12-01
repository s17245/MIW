# MIW_3 zad 1, s17245

import random as r
from math import sqrt

# słownik nazw
zagranie  = {
             '0':'papier', 
             '1':'kamień', 
             '2':'nożyce'
            }    
    
# słownik możliwości    
licznik = { 
              '000' : 3, '001' : 3, '002' : 3, 
              '010' : 3, '011' : 3, '012' : 3, 
              '020' : 3, '021' : 3, '022' : 3, 
              '100' : 3, '101' : 3, '102' : 3, 
              '110' : 3, '111' : 3, '112' : 3, 
              '120' : 3, '121' : 3, '122' : 3,
              '200' : 3, '201' : 3, '202' : 3, 
              '210' : 3, '211' : 3, '212' : 3, 
              '220' : 3, '221' : 3, '222' : 3 
            }

# roztrzygnięcie gry: g gracz, k komputer
def wynik(g, k):
    if (g == k):
        return 'remis'
    elif (g == '0' and k == '1' or g == '1' and k == '2' or g == '2' and k == '0'):
        return 'wygrana'
    else:
        return 'przegrana'


l_wygrane = 0
l_przegrane = 0
l_remis = 0

ost = '33'

liczba_gier = int(input('wprowadź liczbę rund' + '\n'))
liczba_rund = liczba_gier

while(liczba_rund <= liczba_gier):
    
    roll = input('wprowadź:'
                 + '\n' + 'p dla papier'
                 + '\n' + 'k dla kamień'
                 + '\n' + 'n dla nożyce' + '\n')
    
    if   roll == 'p':
        x = '0'
    elif roll == 'k':
        x = '1'
    elif roll == 'n':
        x = '2'

    if(ost[0] == '3'):
        y = str( r.randint(0,2) )
    else:
        p_licznik = licznik[ost + '0']
        k_licznik = licznik[ost + '1']
        n_licznik = licznik[ost + '2']

        suma = p_licznik + k_licznik + n_licznik

        odl = [ p_licznik/suma, k_licznik/suma, 1- (p_licznik/suma) - (k_licznik/suma) ]
        
        rezultat = [ max(odl[2]-odl[1],0), max(odl[0]-odl[2],0), max(odl[1]-odl[0],0) ]
        rezultatnorm = sqrt(rezultat[0]*rezultat[0] + rezultat[1]*rezultat[1] + rezultat[2]*rezultat[2])
        rezultat = [rezultat[0]/rezultatnorm, rezultat[1]/rezultatnorm, 1 - rezultat[0]/rezultatnorm - rezultat[1]/rezultatnorm]

        y = r.uniform(0,1)

        if y <= rezultat[0]:
            y = '0'
        elif y <= rezultat[0] + rezultat[1]:
            y = '1'
        else:
            y = '2'

#aktualizacja słownika
        licznik[ost+x] += 1

    ost = ost[1] + x
# roztrzygnięcie gry
    print ('\n' +'zagrałeś: ' + zagranie[x] 
         + '\n' + 'komputer zagrał: ' + zagranie[y] 
         + '\n' + 'wynik: ', wynik(x,y)
          )
# tabela wyników
    if wynik(x,y) == 'przegrana':
        l_przegrane += 1
    elif wynik(x,y) == 'remis':
        l_remis   += 1
    elif wynik(x,y) == 'wygrana':
        l_wygrane   += 1
        
    liczba_rund -= 1
    
    print('zostało rund: ' , liczba_rund, '\n')
    if(liczba_rund==0):
        break
        
# tabela wyników
print ('\n' + 'wygranych:', l_wygrane, 'przegranych:', l_przegrane, 'remisów:', l_remis)