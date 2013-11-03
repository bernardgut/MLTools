import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
##PARAM##
#of hidden nodes
h1 = 2

#Logistic error for output a2 for a2.len datapoints and n2.len labels
def getError(a2, labels) :
    s = 0
    #all are element-wise operations
    x = np.log(1+np.exp(np.dot(-labels,a2)))
    print 'getError shape ', x.shape
    return np.sum(x)

#transfer funtion
def transferF(a1,a2) :
    return a1*phiF(a2)

#phi
def phiF(x) :
    return 1/(1+np.exp(-x))


#load data
T=np.load('mnist/nTrainingSet.npy')
V=np.load('mnist/nValidationSet.npy')
print 'load success'
print T.shape,' ',V.shape

#display data
#imgplot = plt.imshow(A)
#plt.show()
#np.reshape(A, A.size)




