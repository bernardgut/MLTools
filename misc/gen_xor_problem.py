#Module tu generate XOR problem
import splitter as splitter

import random
import numpy as np

random.seed()
A = list()
L = list()
for i in range(0,1000) :
    x = random.randint(-9,10)
    y = random.randint(-9,10)
    if (x>0  & y >0) :
        t  = -1
    elif (x<=0 and y>0) or (x>0 and y<=0) :
        t = 1
    else :
        t = -1
    print 'test : ',(x,y),' l: ', t
    
    A.append([x,y])
    L.append([t])

A = np.array(A)
L = np.array(L)

#min max
max = np.amax(A)
min = np.amin(A)
#normalize
J = min*np.ones(A.shape)
N = (A-J)/(max-min)

#split data
(T, T_l, V, V_l)=splitter.split(N, L)
print 'Training.training : ', T.shape, ' l : ',T_l.shape, ' ; Training.validation : ',V.shape, ' l : ', V_l.shape, '\n ; writing to disk...'

np.save('../mnist/n_XOR_Training',T)
np.save('../mnist/n_XOR_Validation',V)
np.save('../mnist/n_XOR_Training_labels',T_l)
np.save('../mnist/n_XOR_Validation_labels',V_l)
