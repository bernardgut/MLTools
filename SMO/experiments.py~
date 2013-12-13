import numpy as np
import os, time
import newSMO

#Gaussian function
def gaussian(A,tauGaussian) :
    return np.exp(-tauGaussian*A)

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
        #onesXX = np.asmatrix(np.ones(Y.shape[0])).T
        #onesYY = np.asmatrix(np.ones(X.shape[0])).T
        #A = 0.5*(dxx*onesXX.T) + 0.5*(onesYY*dyy.T) - XYT
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

T_d = list()
T_l = list()
V_d = list()
V_l = list()
for i in range(0,10) :
    T_d.append(np.load("./Data_Label/T_d_"+ str(i) + ".npy"))
    T_l.append(np.load("./Data_Label/T_l_"+ str(i) + ".npy"))
    V_d.append(np.load("./Data_Label/V_d_"+ str(i) + ".npy"))
    V_l.append(np.load("./Data_Label/V_l_"+ str(i) + ".npy"))

#initialization
tau = 10e-8
tauGaussian = 10e-4
C = 10e-2
threshold = 10e-15
while tauGaussian < 10e3 :
    break
    generateAllMatrices(T_d,V_d,tauGaussian)
    while C < 10e4 :
        nbError = 0
        for i in range(0,10) :
            K = np.mat(np.load("./Matrices/K_" + str(tauGaussian) + "_" +str(i) + ".npy"))
            Ktest = np.mat(np.load("./Matrices/Ktest_" + str(tauGaussian) + "_" +str(i) + ".npy"))
            nbError = nbError + newSMO.SMO(np.mat(T_d[i]),np.mat(T_l[i]),np.mat(V_d[i]),np.mat(V_l[i]),tau,tauGaussian,C,threshold,K,Ktest)
            print "i,tauGaussian,C,error",i,tauGaussian,C,nbError
        nbError = nbError / 10
        f = open("./Result/result_" + str(tauGaussian) + "_" +str(C),"w")
        f.write(str(nbError))
        f.write(nbError)
        f.close()
        C = C * 10
    removeOldMatrices()
    tauGaussian = tauGaussian * 10
