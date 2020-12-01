#S17245
import random

lokacja = ['dom', 'stołówka', 'klub', 'wydział']
p_dom = 0
rok = 8766
zmianaCo = int(rok/4)
start = lokacja[0]

def przemieszczenie(miejsce, p):
    if miejsce==lokacja[0]:
        if p>=0 and p <=3:      # dom -> stołówka p=0.4
            return lokacja[1]
        if p>=4 and p <=8:      # dom -> wydział p=0.5
            return lokacja[3]
        if p == 9:              # dom -> dom p=0.1
            return lokacja[0]  
    if miejsce==lokacja[1]:
        if p>=0 and p <=4:      # stołówka -> dom p=0.5
            return lokacja[0]
        if p>=5 and p <=6:      # stołówka -> wydział p=0.2
            return lokacja[3]
        if p>=7 and p <=9:      # stołówka -> klub p=0.3
            return lokacja[2]
    if miejsce==lokacja[2]:
        if p>=0 and p <=9:      # klub -> dom p=1
            return lokacja[0]
    if miejsce==lokacja[3]:    
        if p>=0 and p <=3:      # wydział -> dom p=0.4
            return lokacja[0]
        if p>=4 and p <= 6:     # wydział -> stołówka p=0.3
            return lokacja[1]
        if p>=7 and p <= 9:     # wydział -> klub p=0.3
            return lokacja[2]

for i in range (zmianaCo):
    rand = random.randint(0, 9)
    start = przemieszczenie(start, rand)
    if przemieszczenie(start, rand) == lokacja[0]: 
        p_dom += 1

x = p_dom/zmianaCo  
print("średnie prawdopodobieństwo: ",round(x,5))