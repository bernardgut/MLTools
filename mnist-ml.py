import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#Logistic error for output a2 for a2.len datapoints and n2.len labels
def getError(a2, labels) :
    s = 0
    #all are element-wise operations
    x = np.log(1+np.exp(np.dot(-labels,a2)))
    #print 'getError shape ', x.shape
    return np.sum(x)

#transfer funtion
#def transferF(a1,a2) :
#    return a1*phiF(a2)

#phi
def sigmaF(x) :
    return 1/(1+np.exp(-x))

#Forward pass Perceptron Algorithm
def forwardPass(Xi, W) :
    W1_odd = W[0]
    W1_even = W[1]
    B1_odd = W[2]
    B1_even = W[3]
    W2 = W[4]
    B2 = W[5]
    #Forward pass 
    A1_odd = np.dot(W1_odd, np.transpose(Xi))
    A1_odd = np.add(A1_odd,B1_odd)

    A1_even = np.dot(W1_even, np.transpose(Xi))
    A1_even = np.add(A1_even,B1_odd)

    A2 = np.multiply(A1_odd, sigmaF(A1_even))
    A2 = np.dot(np.transpose(A2),W2)
    A2 = A2 + B2
    return [A1_odd, A1_even, A2]

def backwardPass(A,Ti) :
    A1_odd = A[0]
    A1_even = A[1]
    A2 = A[2]
    #Backward pass
    #layer 2
    R2 = np.add(sigmaF(A2), -Ti)
    dw2_Ei = np.multiply(R2,  np.multiply(A1_odd, sigmaF(A1_even)))
    #layer 1
    R1_odd = np.multiply(R2, np.multiply(W2, sigmaF(A1_even)))
    R1_even = np.multiply(R2, np.multiply(W2, np.multiply(sigmaF(A1_even),sigmaF(-A1_even))))

    dwo1_Ei = np.outer(R1_odd,Xi)
    dwe1_Ei = np.outer(R1_even,Xi)
    return [dwo1_Ei, dwe1_Ei, R1_odd, R1_even, dw2_Ei, R2]  

##PARAMS##
#of hidden nodes
h1 = 2
#step size
rho = 1
#momentum factor
mu = 0

#load data
#X1=np.load('mnist/nTrainingSet.npy')
#X2=np.load('mnist/nValidationSet.npy')
#T = np.load('mnist/labels.npy')
#TODO build map from input to labels
X1 = np.array([[0,0], [0,1], [1,0],[1,1]])
X2 = np.array([[1,0]])
T = np.array([[-1], [1], [1], [-1]])

print 'load success'
print 'Training set X1: ', X1.shape,' ; Valitation set X2: ', X2.shape, ' ; labels set T:', T.shape
#dimensionality
d = X1.shape[1]

#TODO : generate with correct variance
#TODO : early stopping
#Generate initial weights layer 1
W1_odd = np.random.normal(0,1.0,(h1,d))
W1_even = np.random.normal(0,1.0,(h1,d))
B1_odd = np.random.normal(0,1.0,(h1,1))
B1_even = np.random.normal(0,1.0,(h1,1))

#Generate initial weights layer 2 
W2 = np.random.normal(0,1.0,(h1,1))
B2 = np.random.normal(0,1.0)

W = [W1_odd,W1_even,B1_odd,B1_even,W2,B2]

#Labels adjustment
T_t = np.add(T,1)
T_t = np.multiply(T_t,0.5)

DELTA_Wkminone = [0, 0, 0, 0, 0, 0]
DELTA_Wk = [0, 0, 0, 0, 0, 0]
for x in range(1,100) :
    i = 0
    print '=========================================================='
    for Xi in X1 : 
        #select input
        Xi = np.matrix(Xi)
        Ti = T_t[i]
        #Forward pass
        A = forwardPass(Xi,W)
        print 'Label t for Xi : ',Ti
        print 'a2 for given point :', A[2]
        #display error for current point
        E_log = getError(A[2],Ti)
        print 'Logistic Error for given point : ',E_log

        #Backward pass
        GRAD = backwardPass(A, Ti)
        
        DELTA_Wk[0] = np.add(-np.multiply(rho*(1-mu), GRAD[0]),np.multiply(mu,DELTA_Wkminone[0]))
        DELTA_Wk[1] = np.add(-np.multiply(rho*(1-mu), GRAD[1]),np.multiply(mu,DELTA_Wkminone[1]))
        DELTA_Wk[2] = np.add(-np.multiply(rho*(1-mu), GRAD[2]),np.multiply(mu,DELTA_Wkminone[2]))
        DELTA_Wk[3] = np.add(-np.multiply(rho*(1-mu), GRAD[3]),np.multiply(mu,DELTA_Wkminone[3]))
        DELTA_Wk[4] = np.add(-np.multiply(rho*(1-mu), GRAD[4]),np.multiply(mu,DELTA_Wkminone[4]))
        DELTA_Wk[5] = np.add(-np.multiply(rho*(1-mu), GRAD[5]),np.multiply(mu,DELTA_Wkminone[5]))
        
        #apply correction
        W[0] = np.add(W[0], DELTA_Wk[0])
        W[1] = np.add(W[1], DELTA_Wk[1])
        W[2] = np.add(W[2], DELTA_Wk[2])
        W[3] = np.add(W[3], DELTA_Wk[3])
        W[4] = np.add(W[4], DELTA_Wk[4])
        W[5] = np.add(W[5], DELTA_Wk[5])
        
        i=i+1
        #DELTA_Wkminone = DELTA_Wk

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
