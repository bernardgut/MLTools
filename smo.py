import numpy as np

#### AGLG
tau_high = 10e-4
C = 1
tau_low = 10e-15
def select_pair(F, I_low, I_up) : 
    ma = np.amax(F[I_low])
    mi = np.amin(F[I_up])
    indexes = range(0,F.shape[0])
    i_low = [i for i in indexes if i in I_low and F[i,0]==ma]
    i_up = [i for i in indexes if i in I_up and F[i,0]==mi]
    return (i_low[0], i_up[0])

def gaussian(A) :
    return np.exp(-tau_high*A)
    
def initK(X) :
    XXT = (X*X.T)
    d = XXT.diagonal().T
    ones = np.asmatrix(np.ones(d.shape))
    A = 0.5*(d*ones.T) + 0.5*(ones*d.T) - XXT
    return gaussian(A)

#init sets
def indexSets(Alpha) : 
    indexes = range(0,Alpha.shape[0])
    I_0 = [i for i in indexes if 0<Alpha[i,0] and Alpha[i,0]<10]
    I_p = [i for i in indexes if T[i,0]==1 and Alpha[i,0]==0]
    I_m = [i for i in indexes if T[i,0]==-1 and Alpha[i,0]==0]
    return (I_0 + I_p, I_0 + I_m)
    
    
#little xor problem
X = np.matrix('(0,0);(1,0);(0,1);(1,1)')
T = np.matrix('-1;1;1;-1')
#dimensionality
d = X.shape[1]
#size of input
n = X.shape[0]

#initialization
F = -T
K = initK(X)
Alpha = np.asmatrix(np.zeros((n,1)))
(I_low, I_up) = indexSets(Alpha)
print I_low
print I_up
#main loop
while 1 :
    (i,j) = select_pair(F, I_low, I_up)
    print 'i, j :',(i,j)
    if j == -1 :
        break
    sigma = T[i,0]*T[j,0]
    #compute L,H
    w = Alpha[i,0]+sigma*Alpha[j,0]
    IC = 0 
    if sigma == 1: IC = C
    L = max(0,sigma*w-IC)
    IC = 0 
    if sigma == -1 : IC = C
    H = min(C, sigma*w + IC)
    alpha_new_j = 0
    rho = K[i,i] + K[j,j] - 2*K[i,j]
    if rho > tau_low :
        alpha_unc = Alpha[j,0] + (T[j,0]*(F[i,0]-F[j,0])/rho)
        if alpha_unc in range(L,H) : 
           alpha_new_j = alpha_unc
        if alpha_unc < L :
           alpha_new_j = L
        if alpha_unc > H :
           alpha_new_j = H
    else : #Second derivative is negative
        L_i = w - sigma*L
        H_i = w - sigma*H
        #partie insupportable
        v_i = F[i,0] + T[i,0] - Alpha[i,0]*T[i,0]*K[i,i] - Alpha[j,0]*T[j,0]*K[i,j]
        v_j = F[j,0] + T[j,0] - Alpha[i,0]*T[i,0]*K[i,j] - Alpha[j,0]*T[j,0]*K[j,j]
        phi_L = 0.5*(K[i,i]*L_i*L_i+K[j,j]*L*L)+sigma*K[i,j]*L_i*L+T[i,0]*L_i*v_i + T[j,0]*L*v_j - L_i - L #formule insupportable
        phi_H = 0.5*(K[i,i]*H_i*H_i+K[j,j]*H*H)+sigma*K[i,j]*H_i*H+T[i,0]*H_i*v_i + T[j,0]*H*v_j - H_i - H #formule insupportable
        if phi_L > phi_H :
            alpha_new_j = H
        else :
            alpha_new_j
    alpha_new_i = Alpha[i,0] + sigma*(Alpha[j,0] - alpha_new_j)
    F = np.add(F,np.add(T[i,0]*(alpha_new_i-Alpha[i,0])*K[:,i],T[j,0]*(alpha_new_j - Alpha[i,0])*K[:,j]))
    print F
    Alpha[i,0] = alpha_new_i
    Alpha[j,0] = alpha_new_j
    (I_low, I_up) = indexSets(Alpha)
    
