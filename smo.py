import numpy as np
import math

@np.vectorize
def gaussian(a) :
    sigma = 1 #Variance?
    return math.exp(-sigma*a)
    
def initK(X) :
    XXT = (X*X.T)
    d = XXT.diagonal().T
    ones = np.ones(d.shape)
    A = 0.5*(d*ones.T) + 0.5*(ones*d.T) - XXT
    print A
    return gaussian(A)
#little xor problem
X = np.matrix('(0,0);(1,0);(0,1);(1,1)')
t = np.matrix('-1;1;1;-1')
K = initK(X)
print K
    

