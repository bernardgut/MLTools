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
#def transferF(a1,a2) :
#    return a1*phiF(a2)

#phi
def sigmaF(x) :
    return 1/(1+np.exp(-x))


#load data
X1=np.load('mnist/nTrainingSet.npy')
X2=np.load('mnist/nValidationSet.npy')
T = np.load('mnist/labels.npy')

print 'load success'
print 'Training set X1: ', X1.shape,' ; Valitation set X2: ', X2.shape, ' ; labels set T:', T.shape
#dimensionality
d = X1.shape[1]

#TODO : generate with correct variance
#Generate initial weights layer 1
W1_odd = np.random.normal(0,1.0,(h1,d))
W1_even = np.random.normal(0,1.0,(h1,d))
B1_odd = np.random.normal(0,1.0,(h1,1))
B1_even = np.random.normal(0,1.0,(h1,1))

#Generate initial weights layer 2 
W2 = np.random.normal(0,1.0,(h1,1))
B2 = np.random.normal(0,1.0)

#select input
Xi = np.matrix(X1[0])

#Formward pass 
A1_odd = np.dot(W1_odd, np.transpose(Xi))
A1_odd = np.add(A1_odd,B1_odd)

A1_even = np.dot(W1_even, np.transpose(Xi))
A1_even = np.add(A1_even,B1_odd)

A2 = np.multiply(A1_odd, sigmaF(A1_even))
A2 = np.dot(np.transpose(A2),W2)
A2 = A2 + B2

#Labels adjustment
T_t = np.add(T,1)
T_t = np.multiply(T_t,0.5)

#Backward pass
#layer 2
R2 = np.add(sigmaF(A2), -T_t[0])
dw2_Ei = np.multiply(R2,  np.multiply(A1_odd, sigmaF(A1_even)))
#layer 1
R1_odd = np.multiply(R2, np.multiply(W2, sigmaF(A1_even)))
R1_even = np.multiply(R2, np.multiply(W2, np.multiply(sigmaF(A1_even),sigmaF(-A1_even))))
print R1_odd.shape
print R1_even.shape

dwo1_Ei = np.outer(R1_odd,Xi)
dwe1_Ei = np.outer(R1_even,Xi)

#################################
#a = np.array([[1] ,[2] ,[3]])
#print a.shape
#b = np.array([[1 ,2 ,3 ,4 ,5]])
#print b.shape
#dwo1_Ei = np.dot(a,b)
#print dwo1_Ei 
#print W1_odd.shape,"\n",W1_odd
#W1_even = np.array((h1,1))
#b2
#b1_odd = np.array((h1,1))
#b1_even = np.array((h1,1))

#Forward pass
#A1_o = w_o* 
#A1_e =
#A2 =
#display data
#imgplot = plt.imshow(A)
#plt.show()
#np.reshape(A, A.size)




