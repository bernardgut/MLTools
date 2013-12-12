import numpy as np
import random

def split(X,T,i) :
    size = X.shape[0]/10
    n = X.shape[0]
    range(i*size,((i+1)*size))
    if i == 0 :
        V_l = T[range(0,size),:]
        V_d = X[range(0,size),:]
        T_l = T[range(size,n),:]
        T_d = X[range(size,n),:]
    if i == 9 :
        T_l = T[range(0,n-size),:]
        T_d = X[range(0,n-size),:]
        V_l = T[range(n-size,n),:]
        V_d = X[range(n-size,n),:]
    if i < 9 and i > 0 :
        T_l_low = T[range(0,i*size),:]
        T_l_high = T[range((i+1)*size,n),:]
        T_d_low = X[range(0,i*size),:]
        T_d_high = X[range((i+1)*size,n),:]
        T_l = np.concatenate((T_l_low,T_l_high),axis=0)
        T_d = np.concatenate((T_d_low,T_d_high),axis=0)
        V_l = T[range(i*size,((i+1)*size)),:]
        V_d = X[range(i*size,((i+1)*size)),:]
    return (T_d,T_l,V_d,V_l)

#Load big training dataset
A=np.load('../mnist/n_MNIST_Complete49.npy')
#Load labels
L=np.load('../mnist/n_MNIST_Complete49_Label.npy')

#shuffle the data
indexes = range(0, A.shape[0])
random.shuffle(indexes)
indexes = indexes[:100]
A = A[indexes,:]
L = L[indexes,:]
#Save corresponding labels and training set
np.save("./Data_Label/A",A)
np.save("./Data_Label/L",L)
for i in range (0,10) :
    (T_d,T_l,V_d,V_l) = split(A,L,i)
    np.save("./Data_Label/T_d_"+ str(i),T_d)
    np.save("./Data_Label/T_l_"+ str(i),T_l)
    np.save("./Data_Label/V_d_"+ str(i),V_d)
    np.save("./Data_Label/V_l_"+ str(i),V_l)
    
