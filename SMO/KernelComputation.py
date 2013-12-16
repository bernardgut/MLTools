import numpy as np
import os, time

#Gaussian function
def gaussian(A,tauGaussian) :
    return np.exp(-tauGaussian*A)

#Remove matrices for saving space
def removeOldMatrices() :
    dirpath = "./Matrices"
    filelist = [ f for f in os.listdir(dirpath)]
    for f in filelist:
         os.remove(dirpath+"/"+f)

#Initialize the kernel matrix
def initK(X,Y=None,tauGaussian=0.1) :
    if Y == None:
        X = np.mat(X)
        XXT = X * X.T
        dxx = XXT.diagonal().T
        dxx_mat = np.tile(dxx, [1, X.shape[0]])
        A = 0.5 * dxx_mat + 0.5 * dxx_mat.T - XXT
    else:
        X = np.mat(X)
        Y = np.mat(Y)
        XYT = X * Y.T
        XXT = X * X.T
        YYT = Y * Y.T
        dxx = XXT.diagonal().T
        dyy = YYT.diagonal().T
        dxx_mat = np.tile(dxx, [1,Y.shape[0]])
        dyy_mat = np.tile(dyy, [1,X.shape[0]])
        A = 0.5 * dxx_mat + 0.5 * dyy_mat.T - XYT
    return gaussian(A, tauGaussian)

#Generate all Kernel matrix before experiment
def generateAllMatrices(T_d,V_d,tauGaussian) :
    for i in range(0,10) :
        K = initK(T_d[i],tauGaussian=tauGaussian)
        Ktest = initK(T_d[i],V_d[i],tauGaussian=tauGaussian)
        np.save("./Matrices/K_" + str(tauGaussian) + "_" +str(i), K)
        np.save("./Matrices/Ktest_" + str(tauGaussian) + "_" + str(i), Ktest)
        print 'generated matrices for fold #', i
