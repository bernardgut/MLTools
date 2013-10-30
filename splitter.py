import random
import numpy as np

def split(A) :
    random.seed()
    i=0
    j=0
    T = list()
    V= list()
    for x in A:
        if random.randint(0,2)<=1 :
            #training set
            T.append(x)
            #i+=1
        else :
            #validation set
            V.append(x)    
            #j+=1
    return (np.array(T),np.array(V))
