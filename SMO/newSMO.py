import numpy as np
import math

#Select the most violated pair
def select_pair(F, I_low, I_up,cpt,tau) :
    ma = np.amax(F[I_low])
    mi = np.amin(F[I_up])
    if cpt % 20 == 0:
        print math.log(ma - mi,10)
    i_low = np.argwhere(F == ma).flat[0]
    i_up = np.argwhere(F == mi).flat[0]
    #Check for optimality
    if F[i_low,0] <= (F[i_up,0] + 2*tau) :
        return -1, -1
    return i_low, i_up

#init sets
def indexSets(Alpha,T, C) :
    I_0 = np.argwhere(np.logical_and(0 < Alpha[:,0], Alpha[:,0] < C))[:,:,0]
    I_p = np.argwhere(np.logical_or(np.logical_and(T[:,0] == 1, Alpha[:,0] == 0), np.logical_and(T[:,0] == -1, Alpha[:,0] == C)))[:,:,0]
    I_m = np.argwhere(np.logical_or(np.logical_and(T[:,0] == -1, Alpha[:,0] == 0), np.logical_and(T[:,0] == 1, Alpha[:,0] == C)))[:,:,0]
    I_l, I_u = np.vstack([I_0, I_m]), np.vstack([I_0, I_p])
    return (I_l, I_u)
#Compute criterion
def criterion(Alpha, T, K):
    return (0.5*np.sum(np.sum(np.multiply(np.multiply(Alpha * Alpha.T, T * T.T), K), 0), 1) - np.sum(Alpha, 0))

#Get the index of the missclassified element
def getIndexMissclassified(pred,V_l) :
     return (np.argwhere(np.logical_and(pred[:,0] > 0., V_l[:,0] < 0.))[:,:,0], np.argwhere(np.logical_and(pred[:,0] < 0., V_l[:,0] > 0.))[:,:,0])

#Compute prediction
def prediction(Alpha,C,T,Ktest, K) :
    S = np.argwhere(np.logical_and(0 < Alpha[:,0], Alpha[:,0] < C))[:,:,0]
    AT = np.multiply(Alpha, T)
    Y = (AT.T * K).T
    if np.size(S) == 0 :
        b = 0
    else : 
        b = np.sum(T - Y) / float(np.size(S))
    pred = (AT.T * Ktest).T - b
    return pred    

#Compute the number of error
def getError(pred,V_l) :
    return np.sum(np.multiply(pred, V_l) < 0)

#SMO algorithm
def SMO (X,T,tau,tauGaussian,C,threshold,K,iteration) :
    d = X.shape[1]
    n = X.shape[0]
    F = np.mat(-T, dtype=float)
    Alpha = np.mat(np.zeros((n,1)))
    (I_low, I_up) = indexSets(Alpha, T, C)
    BestAlpha = Alpha
    cpt = 0
    while cpt < iteration :
        cpt = cpt + 1
        if cpt % 20 == 0 :
            print "Criterion "+str(criterion(BestAlpha, T, K))
        (i,j) = select_pair(F, I_low, I_up,cpt,tau)
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
        alpha_new_j = 0
        rho = K[i,i] + K[j,j] - 2*K[i,j]
        if rho > threshold :
            alpha_unc = Alpha[j,0] + (T[j,0] * (F[i,0] - F[j,0]) / float(rho))  
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
        F = F + T[i,0] * (alpha_new_i - Alpha[i,0]) * K[:,i] + T[j,0] * (alpha_new_j - Alpha[j,0]) * K[:,j]
        #Update alpha
        Alpha[i,0] = alpha_new_i
        Alpha[j,0] = alpha_new_j
        (I_low, I_up) = indexSets(Alpha,T, C)
    #finalError = prediction(Alpha,C,T,Ktest, K, V_l)
    #print "Error: ",finalError, "criterion of", str(criterion(Alpha, T, K))
    #return finalError
    return Alpha
