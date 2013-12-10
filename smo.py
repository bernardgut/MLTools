import numpy as np
import random
import math

####Parameters
tau = 10e-4
tauGaussian = 0.1
C = 5
threshold = 10e-15
#Select the most violated pair
def select_pair(F, I_low, I_up,cpt) :
    ma = np.amax(F[I_low])
    mi = np.amin(F[I_up])
    if cpt % 100 == 0:
        print math.log(ma - mi,10)
    i_low = np.argwhere(F == ma).flat[0]
    i_up = np.argwhere(F == mi).flat[0]
    #Check for optimality
    if F[i_low,0] <= (F[i_up,0] + 2*tau) :
        return -1, -1
    return i_low, i_up

#Genereate 10-fold set for cross-validation
def split(X,T,i) :
    size = X.shape[0]/10
    n = X.shape[0]
    range(i*size,((i+1)*size))
    if i == 0 :
        V_l = T[range(0,size),:]
        V_d = X[range(0,size),:]
        T_l = T[range(size,n),:]
        T_d = X[range(size,n),:]
    if i == 9 :
        T_l = T[range(0,n-size),:]
        T_d = X[range(0,n-size),:]
        V_l = T[range(n-size,n),:]
        V_d = X[range(n-size,n),:]
    if i < 9 and i > 0 :
        T_l_low = T[range(0,i*size),:]
        T_l_high = T[range((i+1)*size,n),:]
        T_d_low = X[range(0,i*size),:]
        T_d_high = X[range((i+1)*size,n),:]
        T_l = np.concatenate((T_l_low,T_l_high),axis=0)
        T_d = np.concatenate((T_t_low,T_t_high),axis=0)
        V_l = T[range(i*size,((i+1)*size)),:]
        V_d = X[range(i*size,((i+1)*size)),:]
    return (T_d,T_l,V_d,V_l)

#Gaussian function
def gaussian(A) :
    return np.exp(-tauGaussian*A)

#Initialize the kernel matrix 
def initK(X,Y) :
    XYT = np.asmatrix(X)*np.asmatrix(Y.T)
    XXT = np.asmatrix(X)*np.asmatrix(X.T)
    YYT = np.asmatrix(Y)*np.asmatrix(Y.T)
    dxx = XXT.diagonal().T
    dyy = YYT.diagonal().T    
    onesXX = np.asmatrix(np.ones(Y.shape[0])).T
    onesYY = np.asmatrix(np.ones(X.shape[0])).T
    A = 0.5*(dxx*onesXX.T) + 0.5*(onesYY*dyy.T) - XYT
    return gaussian(A)

#init sets
def indexSets(Alpha,T) : 
    indexes = range(0,Alpha.shape[0])
    I_0 = [i for i in indexes if 0<Alpha[i,0] and Alpha[i,0]<C]
    I_p = [i for i in indexes if (T[i,0]==1 and Alpha[i,0]==0) or (T[i,0]==-1 and Alpha[i,0]==C)]
    I_m = [i for i in indexes if (T[i,0]==-1 and Alpha[i,0]==0) or (T[i,0]==1 and Alpha[i,0]==C)]
    I_l, I_u = I_0 + I_m, I_0 + I_p
    return (I_l, I_u)

def criterion(Alpha, T):
    return (0.5*np.sum(np.sum(np.multiply(np.multiply(Alpha * Alpha.T, T * T.T), K), 0), 1) - np.sum(Alpha, 0))

#little xor problem
X = np.matrix('(-1,-1);(1,-1);(-1,1);(1,1)')
V = np.matrix('(-1,-1);(1,-1);(-1,1);(1,1)')
T = np.matrix('(-1);(1);(1);(-1)')
V_l = np.matrix('(-1);(1);(1);(-1)')
#(A,B,C,D) = split(X,T,3)

#Big training dataset
A=np.load('mnist/n_MNIST_Complete49.npy')
indexes = range(0, A.shape[0])
random.shuffle(indexes)
indexes = indexes[:6000]
A = A[indexes,:]
L=np.load('mnist/n_MNIST_Complete49_Label.npy')
L = L[indexes,:]
print "shape A,L"
print A.shape
print L.shape
X,T,V,V_l = split(A,L,0)
print "shape training point, training label, validation point, validation label"
print X.shape
print T.shape
print V.shape
print V_l.shape
#dimensionality
d = X.shape[1]
print "dimensionality", d
#size of input
n = X.shape[0]
print "size of input",n
#initialization
F = np.matrix(-T, dtype=float)
K = initK(X,X)
Ktest = initK(X,V)
Alpha = np.asmatrix(np.zeros((n,1)))
(I_low, I_up) = indexSets(Alpha, T)
cpt = 0

while cpt < 5000 :
    cpt = cpt + 1
    #if cpt % 100 == 0:
        
    (i,j) = select_pair(F, I_low, I_up,cpt)
   # print "most violated pair",(i,j),"cpt : ",cpt
    if j == -1 :
        break
    sigma = T[i,0] * T[j,0]
    #compute L,H
    w = Alpha[i,0] + sigma * Alpha[j,0]
    sigmaw = Alpha[j,0] + sigma * Alpha[i,0]
    IL = 0 if sigma == -1 else C
    IH = 0 if sigma == 1 else C
    L = max(0, sigmaw - IL)
    H = min(C, sigmaw + IH)
    # print "sigma, L,H,w,IL,IH : ",sigma,L,H,w,IL,IH
    alpha_new_j = 0
    rho = K[i,i] + K[j,j] - 2*K[i,j]
    #print "rho", rho
    if rho > threshold :
        alpha_unc = Alpha[j,0] + (T[j,0] * (F[i,0] - F[j,0]) / rho)
        #print "alpha_unc", alpha_unc, L, H, Alpha[j,0]
        if alpha_unc >= L and alpha_unc <= H : 
           alpha_new_j = alpha_unc
        if alpha_unc < L :
           alpha_new_j = L
        if alpha_unc > H :
           alpha_new_j = H
    else : #Second derivative is negative
        L_i = w - sigma * L
        H_i = w - sigma * H
        v_i = F[i,0] + T[i,0] - Alpha[i,0]*T[i,0]*K[i,i] - Alpha[j,0]*T[j,0]*K[i,j]
        v_j = F[j,0] + T[j,0] - Alpha[i,0]*T[i,0]*K[i,j] - Alpha[j,0]*T[j,0]*K[j,j]
        phi_L = 0.5*(K[i,i]*L_i*L_i+K[j,j]*L*L)+sigma*K[i,j]*L_i*L+T[i,0]*L_i*v_i + T[j,0]*L*v_j - L_i - L
        phi_H = 0.5*(K[i,i]*H_i*H_i+K[j,j]*H*H)+sigma*K[i,j]*H_i*H+T[i,0]*H_i*v_i + T[j,0]*H*v_j - H_i - H
        if phi_L > phi_H :
            alpha_new_j = H
        else :
            alpha_new_j = L
    alpha_new_i = Alpha[i,0] + sigma * (Alpha[j,0] - alpha_new_j)
    # print F.shape, T[i,0], (T[i,0] * (alpha_new_i - Alpha[i,0]) * K[:,i]).shape, (T[j,0] * (alpha_new_j - Alpha[j,0]) * K[:,j]).shape
    F = F + T[i,0] * (alpha_new_i - Alpha[i,0]) * K[:,i] + T[j,0] * (alpha_new_j - Alpha[j,0]) * K[:,j]
    #Update alpha
    Alpha[i,0] = alpha_new_i
    Alpha[j,0] = alpha_new_j
    #print criterion
    #constraint = np.sum(np.multiply(T, Alpha))
    #if constraint != 0.0: raise Exception('Whaaaaat!? ' + str(constraint))
    #print "criterion", criterion(Alpha, T), "constraint", constraint
    (I_low, I_up) = indexSets(Alpha,T)

print "iterations", cpt
#Compute W
#W = np.dot(np.multiply(Alpha[:,0],T[:,0]).T, X).T 
#Compute b
#print "Alpha"
#print Alpha
#print "T"
#print T
S = [i for i in range(0,Alpha.shape[0]) if Alpha[i,0]>0 or Alpha[i,0]<C]
AT = np.multiply(Alpha, T)
Y = (AT.T * K).T
b = np.sum(T - Y) / np.size(S)
pred = (AT.T * Ktest).T - b
print "Correclty classified: ",np.sum(np.multiply(pred, V_l) > 0), " at count ", cpt, "criterion of", criterion(Alpha, T)

