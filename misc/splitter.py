import random
import numpy as np

def split(A, L) :
    random.seed()
    i=0
    j=0
    T = list()
    V = list()
    T_l = list()
    V_l = list()
    for i in range(0,A.shape[0]):
        if random.randint(0,2)<=1 :
            #training set
            T.append(A[i])
            T_l.append(L[i])
        else :
            #validation set
            V.append(A[i])    
            V_l.append(L[i])
    return (np.array(T), np.array(T_l), np.array(V), np.array(V_l))
