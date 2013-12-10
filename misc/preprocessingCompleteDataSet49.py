#Normalizer module
import load_mnist as loader
import splitter as splitter

import numpy as np

#Normalize dataset
def normalize(A):
    #min max
    amax = np.amax(A)
    amin = np.amin(A)
    #normalize
    J = amin*np.ones(A.shape)
    N = (A-J)/(amax-amin)
    return N
    
#SAME for second dataset
#load data
(A,L)=loader.loadTrainingSet_49()
print 'dataset 4-9 : success'
print A.shape
#normalize
N = normalize(A)
print '4-9 : Complete : ', N.shape,L.shape
#save to disk
np.save('../mnist/n_MNIST_Complete49',N)
np.save('../mnist/n_MNIST_Complete49_Label',L)
