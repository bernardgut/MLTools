###############################################################################
# mlp.py
# Multi-Layer Perceptron Implementation (2 hidden layers)
#
# Bernard Gutermann, Vincent Robert - 06.12.2013
###############################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import os

#### TODO #####
    #TODO : stats
    #TODO : exemple of overfitting
        
##############

##PROG PARAMETERS##
_debug = 0
##ALGO PARAMETERS##
#no of hidden nodes [1:inf]
h1 = 15
#step size [0:1]
rho = 0.01
#momentum factor [0:1]
mu = 0.6

#Algorithm parameters initialisation
def init(_h1, _rho, _mu, debug) :
    global h1
    global rho 
    global mu 
    global _debug
    h1 = _h1
    rho = _rho
    mu = _mu
    _debug = debug

#Labels adjustment
def adjustLabels(T):
    T_t = np.add(T,1)
    T_t = np.multiply(T_t,0.5)
    return T_t.astype(int)

#reshuffle
def reshuffle(X,L) : 
    indexes = range(0,X.shape[0])
    random.shuffle(indexes)
    S = X[indexes]
    T = L[indexes]
    return (S, T)
    
#Logistic error for output a2 for a2.len datapoints and n2.len labels
def getError(a2, labels) :
    #all are element-wise operations
    if labels == 0 : labels = -1
    x = np.log1p(np.exp(np.multiply(-labels,a2)))
    #print 'getError shape ', x.shape
    return x.item()

#transfer funtion
#def transferF(a1,a2) :
#    return a1*phiF(a2)

#sigma function
def sigmaF(x) :
    return np.divide(1 , np.add(1, np.exp(-x)))

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
    A2 = np.add(A2, B2)
    
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
    R1_even = np.multiply(R2, np.multiply(A1_odd, np.multiply(W2, np.multiply(sigmaF(A1_even),sigmaF(-A1_even)))))

    dwo1_Ei = np.outer(R1_odd,Xi)
    dwe1_Ei = np.outer(R1_even,Xi)
    return [dwo1_Ei, dwe1_Ei, R1_odd, R1_even, dw2_Ei, R2]  

#validate
def mlp_validation(X2, T2, W) :
    
    count = 0
    E = 0
    for i in range(0,X2.shape[0]) :
            xi = np.matrix(X2[i])
            #Forward pass, A - activators :[A1_odd, A1_even, A2]
            A = forwardPass(xi,W)
            #display error for current point
            E_log = getError(A[2],  T2[i])
            
            #count mistaken classements
            if (A[2] < 0 and T2[i]==1) or (A[2] > 0 and T2[i]==0) : count = count+1
            if _debug :
                #print 'descision for : ', xi
                print 'Label t for Xi : ',  T2[i]
                print 'a2 for given point :', A[2]
                print 'Logistic Error for given point : ',E_log
            
            E = E + E_log
    return (E/X2.shape[0], count)
            
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
        
        #Generate initial weights layer 2 
        W2 = np.matrix(np.random.normal(0,np.sqrt(1.0/d),(h1,1)))
        B2 = np.matrix(np.random.normal(0,np.sqrt(1.0/d)))

        W = [W1_odd,W1_even,B1_odd,B1_even,W2,B2]
    
    #Deltas W :
    #correction at iteration k-1
    DW_kminone = [0, 0, 0, 0, 0, 0]
    #correction at iteration k
    DW_k = [0, 0, 0, 0, 0, 0]

    E = 0     
    count = 0    
    for i in range(0,X1.shape[0]) :
        xi = np.matrix(X1[i])
        #Forward pass, A - activators :[A1_odd, A1_even, A2]
        A = forwardPass(xi,W)
        #display error for current point
        E_log = getError(A[2],  T1[i])
        E = E + E_log
        #count mistaken classements
        if ((A[2] < 0) and (T1[i]==1)) or ((A[2] > 0) and (T1[i]==0)) : count = count+1
            
        #Backward pass GRAD : Residues : GRAD = [dwo1_Ei, dwe1_Ei, R1_odd, R1_even, dw2_Ei, R2]   
        GRAD = backwardPass(A, W[4], xi, T1[i])

        #Gradient descent
        DW_k[0] = np.add(-np.multiply(rho*(1-mu), GRAD[0]),np.multiply(mu,DW_kminone[0]))
        DW_k[1] = np.add(-np.multiply(rho*(1-mu), GRAD[1]),np.multiply(mu,DW_kminone[1]))
        DW_k[2] = np.add(-np.multiply(rho*(1-mu), GRAD[2]),np.multiply(mu,DW_kminone[2]))
        DW_k[3] = np.add(-np.multiply(rho*(1-mu), GRAD[3]),np.multiply(mu,DW_kminone[3]))
        DW_k[4] = np.add(-np.multiply(rho*(1-mu), GRAD[4]),np.multiply(mu,DW_kminone[4]))
        DW_k[5] = np.add(-np.multiply(rho*(1-mu), GRAD[5]),np.multiply(mu,DW_kminone[5]))
        
        #apply correction : W = [W1_odd,W1_even,B1_odd,B1_even,W2,B2]
        W[0] = np.add(W[0], DW_k[0])
        W[1] = np.add(W[1], DW_k[1])
        W[2] = np.add(W[2], DW_k[2])
        W[3] = np.add(W[3], DW_k[3])
        W[4] = np.add(W[4], DW_k[4])
        W[5] = np.add(W[5], DW_k[5])
        
        DW_kminone = DW_k
        
        if _debug :
            #print 'descision for : ', xi
            print 'Label t for Xi : ',  T1[i]
            print 'a2 for given point :', A[2]
            print 'Logistic Error for given point : ',E_log
        
        
    #end train phase for epoch
    return (W, E/X1.shape[0], count)

#################################

###RUN SCRIPT###
def run(_h1=15, _eta=0.01, _mu=0.6,run=0, debug=0) : 
    #alogorithm param init
    init(_h1, _eta, _mu, debug)
    
    #load data
    T1_d=np.load('mnist/n_MNIST_Training35.npy')
    T1_l=np.load('mnist/n_MNIST_Training_labels35.npy')
    V1_d=np.load('mnist/n_MNIST_Validation35.npy')
    V1_l=np.load('mnist/n_MNIST_Validation_labels35.npy')

    T2_d=np.load('mnist/n_MNIST_Training49.npy')
    T2_l=np.load('mnist/n_MNIST_Training_labels49.npy')
    V2_d=np.load('mnist/n_MNIST_Validation49.npy')
    V2_l=np.load('mnist/n_MNIST_Validation_labels49.npy')

    #adjust labels to [0-1] (T tilda)
    T1_l=adjustLabels(T1_l)
    V1_l=adjustLabels(V1_l)
    T2_l=adjustLabels(T2_l)
    V2_l=adjustLabels(V2_l)

    print 'load success : '
    print '3-5 : Training : ', T1_d.shape, ' l : ',T1_l.shape, ' ; Validation : ',V1_d.shape, ' l : ',V1_l.shape
    print '4-9 : Training : ', T2_d.shape, ' l : ',T2_l.shape, ' ; Validation : ',V2_d.shape, ' l : ', V2_l.shape
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

    print 'begining run ',run,' with  parameters :'
    print 'step size: \t\trho\t= ', rho
    print 'momentum factor: \tmu\t= ', mu
    print 'Total # hidden layers: \t|A|\t= ', 2*h1 

    #BEGIN
    W = 0
    W_min=0
    E_min = 999999
    W_min2=0
    E_min2 = 999999
    missed_min = 99999
    missed_min2 = 99999

    Errors_train = list()
    Errors_val = list()
    MissedList_train = list()
    MissedList_val = list()

    es_count = 0
    epoch = 0
    early_s = False
    while epoch<30 : #and early_s = False
        #randomize the order in which the data is read
        (T1_d,T1_l) = reshuffle(T1_d,T1_l)
        (V1_d,V1_l) = reshuffle(V1_d,V1_l)
        #training and validation
        if _debug : print '====================== Epoch ',epoch,': training ========================='
        (W, E_train, missed_train) = mlp_train(T1_d,T1_l,W)
        Errors_train.append(E_train)
        MissedList_train.append(missed_train)
        
        if _debug : print '====================== Epoch ',epoch,': validating ======================='
        (E_val, missed_val) = mlp_validation(V1_d,V1_l,W)
        Errors_val.append(E_val)
        MissedList_val.append(missed_val)
        
        #print info
        print 'epoch ',epoch,' , train error E =', E_train,' mistakes : ',missed_train
        print 'epoch ',epoch,' , validation error E =', E_val,' - delta :', E_val-E_min,' mistakes : ',missed_val
        
        #early stopping, only after the first epoch, it can be that you go in the wrond dir at start
        #we allow the algorithm to go up sparsely, in a moving average fashion
        #note that we want the algorithm to make as few mistakes as possible, so we count only when the number of mistakes goes up. this allows for the error to go up as long as there are less mistakes. Experimentally this was better (report)
        
        if E_val-E_min>0 and missed_val>missed_min and epoch>0 : 
            es_count=es_count+1
        elif E_val-E_min<0 :
            W_min = W
            E_min = E_val
            missed_min = missed_val
            if es_count>0 : es_count = es_count - 1 
        elif missed_val<missed_min2 :
            missed_min2=missed_val
            W_min2 = W
            E_min2 = E_val
            if es_count>0 : es_count = es_count - 1 
        elif missed_val==missed_min2 :
            if E_val<E_min2 :
                W_min2 = W
                E_min2 = E_val
                if es_count>0 : es_count = es_count - 1 
        if es_count >= 5 : early_s = True 
        #go to the next epoch
        epoch = epoch + 1

    #show and save results
    print 'best validation error (non-normalized) : ', E_min,' ; with mistakes : ', missed_min
    print 'best validation error (non-normalized) with respect to mistakes : ',E_min2,' ; with mistakes : ', missed_min2
    
    directory='h'+str(h1)+'R'+str(rho)+'M'+str(mu)
    directory='results/mlp/'+directory
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    np.save(directory+'/ET_'+str(run), np.array(Errors_train))
    np.save(directory+'/EV_'+str(run), np.array(Errors_val))
    np.save(directory+'/MT_'+str(run), np.array(MissedList_train))
    np.save(directory+'/MV_'+str(run), np.array(MissedList_val))
    
###########################################################################
#run()


