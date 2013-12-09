from numpy import *
import scipy.io

def loadTrainingSet_35():
    d = scipy.io.loadmat('../mnist/mp_3-5_data.mat') # corresponding MAT file
    data = d['Xtrain']    # Xtest for test data
    labels = d['Ytrain']  # Ytest for test labels
    print 'Finished loading',data.shape[0],'datapoints'
    return (data, labels)
    
def loadTestSet_35():
    d = scipy.io.loadmat('../mnist/mp_3-5_data.mat') # corresponding MAT file
    data = d['Xtest']    # Xtest for test data
    labels = d['Ytest']  # Ytest for test labels
    print 'Finished loading',data.shape[0],'datapoints'
    return (data, labels)
    
def loadTrainingSet_49():
    d = scipy.io.loadmat('../mnist/mp_4-9_data.mat') # corresponding MAT file
    data = d['Xtrain']    # Xtest for test data
    labels = d['Ytrain']  # Ytest for test labels
    print '\nFinished loading',data.shape[0],'datapoints'
    return (data, labels)
    
def loadTestSet_49():
    d = scipy.io.loadmat('../mnist/mp_4-9_data.mat') # corresponding MAT file
    data = d['Xtest']    # Xtest for test data
    labels = d['Ytest']  # Ytest for test labels
    print '\nFinished loading',data.shape[0],'datapoints'
    return (data, labels)
