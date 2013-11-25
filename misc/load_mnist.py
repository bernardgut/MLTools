from numpy import *
import scipy.io

def loadTrainingSet():
    d = scipy.io.loadmat('../mnist/mp_3-5_data.mat') # corresponding MAT file
    data = d['Xtrain']    # Xtest for test data
    labels = d['Ytrain']  # Ytest for test labels
    print 'Finished loading',data.shape[0],'datapoints'
    return (data, labels)
