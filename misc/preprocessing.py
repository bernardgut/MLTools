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
    
#load data
(A,L)=loader.loadTrainingSet_35()
print 'dataset 3-5 : success'
print A.shape
#normalize
N = normalize(A)
#split data
(T, T_l, V, V_l)=splitter.split(N, L)
print '3-5 : Training.training : ', T.shape, ' l : ',T_l.shape, ' ; Training.validation : ',V.shape, ' l : ', 
#save to disk
V_l.shape, '\n ; writing to disk...'
np.save('../mnist/n_MNIST_Training35',T)
np.save('../mnist/n_MNIST_Validation35',V)
np.save('../mnist/n_MNIST_Training_labels35',T_l)
np.save('../mnist/n_MNIST_Validation_labels35',V_l)

#SAME for second dataset
#load data
(A,L)=loader.loadTrainingSet_49()
print 'dataset 4-9 : success'
print A.shape
#normalize
N = normalize(A)
#split data
(T, T_l, V, V_l)=splitter.split(N, L)
print '4-9 : Training.training : ', T.shape, ' l : ',T_l.shape, ' ; Training.validation : ',V.shape, ' l : ', 
#save to disk
V_l.shape, '\n ; writing to disk...'
np.save('../mnist/n_MNIST_Training49',T)
np.save('../mnist/n_MNIST_Validation49',V)
np.save('../mnist/n_MNIST_Training_labels49',T_l)
np.save('../mnist/n_MNIST_Validation_labels49',V_l)
