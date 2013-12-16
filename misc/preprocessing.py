#Normalizer module
import load_mnist as loader
import splitter as splitter

import numpy as np

#Normalize dataset
def normalize(A, amax=0, amin=0):
    #min max
    if amax==0 and amin ==0 :
        amax = np.amax(A)
        amin = np.amin(A)
    
    #normalize
    J = amin*np.ones(A.shape)
    N = (A-J)/(amax-amin)
    return N, amax, amin
    
#load data
(A,L)=loader.loadTrainingSet_35()
print 'dataset 3-5 : Train : success'
print A.shape
#normalize
N, tmax, tmin = normalize(A)
#split data
(T, T_l, V, V_l)=splitter.split(N, L)
print '3-5 : Training.training : ', T.shape, ' l : ',T_l.shape, ' ; Training.validation : ',V.shape, ' l : ', ' max :',tmax,' min', tmin 
#save to disk
np.save('../mnist/n_MNIST_Training35',T)
np.save('../mnist/n_MNIST_Validation35',V)
np.save('../mnist/n_MNIST_Training_labels35',T_l)
np.save('../mnist/n_MNIST_Validation_labels35',V_l)

#load test data
(A,L)=loader.loadTestSet_35()
print 'dataset 3-5 : Test : success'
print A.shape
#normalize with the same min,max as training set
N,_,__ = normalize(A, tmax, tmin)
#split data
print '3-5 :Test set : ', N.shape, ' l : ',L.shape, ' max :',tmax,' min', tmin
#save to disk
np.save('../mnist/n_MNIST_Test35',N)
np.save('../mnist/n_MNIST_Test_labels35',L)

#SAME for second dataset
#load data
(A,L)=loader.loadTrainingSet_49()
print 'dataset 4-9 : success'
print A.shape
#normalize
N, tmax, tmin = normalize(A)
#split data
(T, T_l, V, V_l)=splitter.split(N, L)
print '4-9 : Training.training : ', T.shape, ' l : ',T_l.shape, ' ; Training.validation : ',V.shape, ' l : ', ' max :',tmax,' min', tmin 
#save to disk
np.save('../mnist/n_MNIST_Training49',T)
np.save('../mnist/n_MNIST_Validation49',V)
np.save('../mnist/n_MNIST_Training_labels49',T_l)
np.save('../mnist/n_MNIST_Validation_labels49',V_l)

#load test data
(A,L)=loader.loadTestSet_49()
print 'dataset 4-9 : Test : success'
print A.shape
#normalize with the same min,max as training set
N,_,__ = normalize(A, tmax, tmin)
#split data
print '4-9 :Test set : ', N.shape, ' l : ',L.shape,' max :',tmax,' min', tmin
#save to disk

np.save('../mnist/n_MNIST_Test49',N)
np.save('../mnist/n_MNIST_Test_labels49',L)

