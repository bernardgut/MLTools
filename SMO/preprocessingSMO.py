import numpy as np
import random
import split10fold
#Load big training dataset
A=np.load('../mnist/n_MNIST_Complete49.npy')
#Load labels
L=np.load('../mnist/n_MNIST_Complete49_Label.npy')

#shuffle the data
indexes = range(0, A.shape[0])
random.shuffle(indexes)
indexes = indexes[:1000]
A = A[indexes,:]
L = L[indexes,:]
#Save corresponding labels and training set
np.save("./Data_Label/A",A)
np.save("./Data_Label/L",L)
for i in range (0,10) :
    T_d,T_l,V_d,V_l = split10fold.split(A,L,i)
    np.save("./Data_Label/T_d_"+ str(i),T_d)
    np.save("./Data_Label/T_l_"+ str(i),T_l)
    np.save("./Data_Label/V_d_"+ str(i),V_d)
    np.save("./Data_Label/V_l_"+ str(i),V_l)
    
