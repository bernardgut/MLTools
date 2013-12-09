import numpy as np

####Parameters
tau = 10e-4
tauGaussian = 0.5
C = 0.01
threshold = 10e-15
#Select the most violated pair
def select_pair(F, I_low, I_up) : 
    ma = np.amax(F[I_low])
    mi = np.amin(F[I_up])
    indexes = range(0,F.shape[0])
    i_low = [i for i in indexes if i in I_low and F[i,0]==ma]
    i_up = [i for i in indexes if i in I_up and F[i,0]==mi]
    #Check for optimality
    if F[i_low[0]] <= (F[I_up[0]]+2*tau) :
        i_low[0] = -1
        i_up[0] = -1
    return (i_low[0], i_up[0])

#Genereate 10-fold set for cross-validation
def split(X,T,i) :
    size = X.shape[0]/10
    n = X.shape[0]
    range(i*size,((i+1)*size))
    if i == 0 :
        V_l = X[range(0,size),:]
        V_t = T[range(0,size),:]
        T_l = X[range(size,n),:]
        T_t = T[range(size,n),:]
    if i == 9 :
        T_l = X[range(0,n-size),:]
        T_t = T[range(0,n-size),:]
        V_l = X[range(n-size,n),:]
        V_t = T[range(n-size,n),:]
    if i < 9 and i > 0 :
        T_l_low = X[range(0,i*size),:]
        T_l_high = X[range((i+1)*size,n),:]
        T_t_low = T[range(0,i*size),:]
        T_t_high = T[range((i+1)*size,n),:]
        T_l = np.concatenate((T_l_low,T_l_high),axis=0)
        T_t = np.concatenate((T_t_low,T_t_high),axis=0)
        V_l = X[range(i*size,((i+1)*size)),:]
        V_t = T[range(i*size,((i+1)*size)),:]
    return (T_t,T_l,V_t,V_l)
#Gaussian function
def gaussian(A) :
    return np.exp(-tauGaussian*A)
#Initialize the kernel matrix 
def initK(X,Y) :
    XYT = (np.asmatrix(X)*np.asmatrix(Y.T))
    XXT = np.asmatrix(X)*np.asmatrix(X.T)
    YYT = np.asmatrix(Y)*np.asmatrix(Y.T)
    dxx = XXT.diagonal().T
    dyy = YYT.diagonal().T
    onesXX = np.asmatrix(np.ones(Y.shape[0])).T
    onesYY = np.asmatrix(np.ones(X.shape[0])).T
    A = 0.5*(dxx*onesXX.T) + 0.5*(onesYY*dyy.T) - XYT
    return gaussian(A)

#init sets
def indexSets(Alpha) : 
    indexes = range(0,Alpha.shape[0])
    I_0 = [i for i in indexes if 0<Alpha[i,0] and Alpha[i,0]<C]
    I_p = [i for i in indexes if (T[i,0]==1 and Alpha[i,0]==0) or (T[i,0]==-1 and Alpha[i,0]==C)]
    I_m = [i for i in indexes if (T[i,0]==-1 and Alpha[i,0]==0) or (T[i,0]==1 and Alpha[i,0]==C)]
    return (I_0 + I_p, I_0 + I_m)

def criterion(Alpha, T):
    return (0.5*np.sum(np.sum(np.multiply(Alpha * Alpha.T, T * T.T, K), 0), 1) - np.sum(Alpha, 0)).flat[0]
    
#little xor problem
X = np.matrix('(-1,-1);(1,-1);(-1,1);(1,1)')
V = np.matrix('(-1,-1);(1,-1);(-1,1);(1,1)')
T = np.matrix('(-1);(1);(1);-(1)')
#(A,B,C,D) = split(X,T,3)

#Big training dataset
#X=np.load('mnist/n_MNIST_Training.npy')[:100,:]
#T=np.load('mnist/n_MNIST_Training_labels.npy')[:100,:]

#Big validation dataset
#X2=np.load('mnist/n_MNIST_Validation.npy')
#T2=np.load('mnist/n_MNIST_Validation_labels.npy')

#dimensionality
d = X.shape[1]
#size of input
n = X.shape[0]

#initialization
F = -T
K = initK(X,X)
Ktest = initK(X,V)
Alpha = np.asmatrix(np.zeros((n,1)))
(I_up, I_low) = indexSets(Alpha)
##print I_low
##print I_up
#main loop
#print K
while 1 :
    (i,j) = select_pair(F, I_low, I_up)
    print (i,j)    
    if j == -1 :
        break
    sigma = T[i,0]*T[j,0]
    #compute L,H
    w = Alpha[i,0]+sigma*Alpha[j,0]
    IL = 0
    IH = 0 
    if sigma == 1: 
        IL = C
    else:
        IH = C
    L = max(0,w-IL)
    H = min(C, w + IH)
   # print "sigma, L,H,w,IL,IH : ",sigma,L,H,w,IL,IH
    alpha_new_j = 0
    rho = K[i,i] + K[j,j] - 2*K[i,j]
    if rho > threshold :
        alpha_unc = Alpha[j,0] + (T[j,0]*(F[i,0]-F[j,0])/rho)
        if alpha_unc >= L and alpha_unc <= H : 
           alpha_new_j = alpha_unc
        if alpha_unc < L :
           alpha_new_j = L
        if alpha_unc > H :
           alpha_new_j = H
    else : #Second derivative is negative
        L_i = w - sigma*L
        H_i = w - sigma*H
        v_i = F[i,0] + T[i,0] - Alpha[i,0]*T[i,0]*K[i,i] - Alpha[j,0]*T[j,0]*K[i,j]
        v_j = F[j,0] + T[j,0] - Alpha[i,0]*T[i,0]*K[i,j] - Alpha[j,0]*T[j,0]*K[j,j]
        phi_L = 0.5*(K[i,i]*L_i*L_i+K[j,j]*L*L)+sigma*K[i,j]*L_i*L+T[i,0]*L_i*v_i + T[j,0]*L*v_j - L_i - L
        phi_H = 0.5*(K[i,i]*H_i*H_i+K[j,j]*H*H)+sigma*K[i,j]*H_i*H+T[i,0]*H_i*v_i + T[j,0]*H*v_j - H_i - H
        if phi_L > phi_H :
            alpha_new_j = H
        else :
            alpha_new_j = L
    alpha_new_i = Alpha[i,0] + sigma*(Alpha[j,0] - alpha_new_j)
    F = np.add(F,np.add(T[i,0]*(alpha_new_i-Alpha[i,0])*K[:,i],T[j,0]*(alpha_new_j - Alpha[j,0])*K[:,j]))
    #Update alpha
    Alpha[i,0] = alpha_new_i
    Alpha[j,0] = alpha_new_j
    print "criterion", criterion(Alpha, T)
    #Check constraint must be equal to 0
    print np.sum(np.multiply(T, Alpha))
    (I_up, I_low) = indexSets(Alpha)
print "Alpha"
print Alpha
#Compute W
#W = np.dot(np.multiply(Alpha[:,0],T[:,0]).T, X).T 

#Compute b
S = [i for i in range(0,Alpha.shape[0]) if Alpha[i,0]>=0 or Alpha[i,0]<=C]
Y = -np.dot(np.multiply(Alpha[:,0],T[:,0]).T,K).T
b = np.sum(np.add(T,Y))/np.size(S)
#Prediction

print np.dot(np.multiply(Alpha[:,0],T[:,0]).T,Ktest[:,2]) - b
#print "W"
#print W
#print "b"
#print b
#print "X"
#print X
#print np.dot(W.T,X.T) - b
