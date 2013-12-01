import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

#### TODO #####
    #TODO : generate with correct variance
    #TODO : early stopping

##############

##PROG PARAMETERS##
_debug = 0
##ALGO PARAMETERS##
#no of hidden nodes [1:inf]
h1 = 10
#step size [0:1]
rho = 0.01
#momentum factor [0:1]
mu = 0.3

#validate
def mlp_validation(X2, T2, W) :
    #Labels adjustment
    T_t = np.add(T2,1)
    T_t = np.multiply(T_t,0.5)
    
    E = 0
    if _debug : print '=====================validation for epoch======================='
    for i in range(0,X2.shape[0]) :
            xi = np.matrix(X2[i])
            #Forward pass, A - activators :[A1_odd, A1_even, A2]
            A = forwardPass(xi,W)
            #display error for current point
            E_log = getError(A[2],  T_t[i])
            
            if _debug :
                #print 'descision for : ', xi
                print 'Label t for Xi : ',  T_t[i]
                print 'a2 for given point :', A[2]
                print 'Logistic Error for given point : ',E_log
            
            E = E + E_log
    return E #/X2.shape[0]
            
##train##
def mlp_train(X1, T1, W=0) :
    #dimensionality
    d = X1.shape[1]
    if W==0 : 
        #Generate initial weights layer 1
        W1_odd = np.matrix(np.random.normal(0,np.sqrt(1.0/d),(h1,d)))
        W1_even = np.matrix(np.random.normal(0,np.sqrt(1.0/d),(h1,d)))
        B1_odd = np.matrix(np.random.normal(0,np.sqrt(1.0/d),(h1,1)))
        B1_even = np.matrix(np.random.normal(0,np.sqrt(1.0/d),(h1,1)))
        
        print 'standard deviation :', 1.0/d
        #Generate initial weights layer 2 
        W2 = np.matrix(np.random.normal(0,np.sqrt(1.0/d),(h1,1)))
        B2 = np.matrix(np.random.normal(0,np.sqrt(1.0/d)))

        W = [W1_odd,W1_even,B1_odd,B1_even,W2,B2]

    #Labels adjustment
    T_t = np.add(T1,1)
    T_t = np.multiply(T_t,0.5)
    
    #Deltas W :
    #correction at iteration k-1
    DW_kminone = [0, 0, 0, 0, 0, 0]
    #correction at iteration k
    DW_k = [0, 0, 0, 0, 0, 0]
    #max_error = 0
    #for epoch in range(0,1000) :
    if _debug : print '======================train epoch=========================' 
    for i in range(0,X1.shape[0]) :
        xi = np.matrix(X1[i])
        #Forward pass, A - activators :[A1_odd, A1_even, A2]
        A = forwardPass(xi,W)
        #display error for current point
        E_log = getError(A[2],  T_t[i])
        
        if _debug :
            #print 'descision for : ', xi
            print 'Label t for Xi : ',  T_t[i]
            print 'a2 for given point :', A[2]
            print 'Logistic Error for given point : ',E_log
        
        #Backward pass GRAD : Residues : [dwo1_Ei, dwe1_Ei, R1_odd, R1_even, dw2_Ei, R2]  
        GRAD = backwardPass(A, W[4], xi, T_t[i])
        #error = test_gradient(GRAD, W, d, h1)
       	#if error > max_error :
		#	max_error = error

        #Gradient descent
        DW_k[0] = np.add(-np.multiply(rho*(1-mu), GRAD[0]),np.multiply(mu,DW_kminone[0]))
        DW_k[1] = np.add(-np.multiply(rho*(1-mu), GRAD[1]),np.multiply(mu,DW_kminone[1]))
        DW_k[2] = np.add(-np.multiply(rho*(1-mu), GRAD[2]),np.multiply(mu,DW_kminone[2]))
        DW_k[3] = np.add(-np.multiply(rho*(1-mu), GRAD[3]),np.multiply(mu,DW_kminone[3]))
        DW_k[4] = np.add(-np.multiply(rho*(1-mu), GRAD[4]),np.multiply(mu,DW_kminone[4]))
        DW_k[5] = np.add(-np.multiply(rho*(1-mu), GRAD[5]),np.multiply(mu,DW_kminone[5]))
        
        #apply correction
        W[0] = np.add(W[0], DW_k[0])
        W[1] = np.add(W[1], DW_k[1])
        W[2] = np.add(W[2], DW_k[2])
        W[3] = np.add(W[3], DW_k[3])
        W[4] = np.add(W[4], DW_k[4])
        W[5] = np.add(W[5], DW_k[5])
    #end train phase for epoch
    #print 'Max error in gradient : ', max_error
    return W

#reshuffle
def reshuffle(X,L) : 
    indexes = range(0,X.shape[0])
    random.shuffle(indexes)
    S = X[indexes[:]]
    T = L[indexes[:]]
    return (S, T)
    
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

#sigma function
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

#Backward pass Percepton Algorithm
def backwardPass(A,W2,Xi,Ti) :
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




#Forward pass
#A1_o = w_o* 
#A1_e =
#A2 =
#display data
#imgplot = plt.imshow(A)
#plt.show()
#np.reshape(A, A.size)

#################################

###RUN SCRIPT###
#load data
X1=np.load('mnist/n_MNIST_Training.npy')
T1=np.load('mnist/n_MNIST_Training_labels.npy')
X2=np.load('mnist/n_MNIST_Validation.npy')
T2=np.load('mnist/n_MNIST_Validation_labels.npy')
#T = np.load('mnist/labels.npy')

#xor problem big
#X1 = np.load('mnist/n_XOR_Training.npy')
#X2 = np.load('mnist/n_XOR_Validation.npy')
#T1 = np.load('mnist/n_XOR_Training_labels.npy')
#T2 = np.load('mnist/n_XOR_Validation_labels.npy')

#xor proplem small
#X1 = np.array([[0,0],[1,0],[0,1],[1,1]])
#T1 = np.array([[-1],[1],[1],[-1]])
#X2 = np.array([[0,0],[1,0],[0,1],[1,1]])
#T2 = np.array([[-1],[1],[1],[-1]])
#print 'load success'
#print 'Training set X1: ', X1.shape,' ; Valitation set X2: '#, X2.shape, ' ; labels set T1:', T1.shape, ' ; labels set T2:', T2.shape

W = 0
Emin = 999999
count = 0
epoch = 0
early_s = 0
while epoch<100 and not early_s : 
    #(X1,T1) = reshuffle(X1,T1)
    W = mlp_train(X1,T1,W)
    E = mlp_validation(X2,T2,W)
    #print info
    print 'epoch ',epoch,' , validation error E =', E,' - delta :', E-Emin
    
    #early stopping, only after the first 100 epochs, it can be that you go in the wrond dir at start
    if E>Emin and epoch>50 : 
        Wmin = W
        count=count+1
    else :
        Emin = E
        if count>0 : count = count-1
    if count >10 : early_s = 1 
    epoch = epoch + 1
    
    #convergence test
