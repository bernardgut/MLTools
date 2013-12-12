import numpy as np
import os
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
def initK(X,Y,tauGaussian) :
    XYT = np.asmatrix(X)*np.asmatrix(Y.T)
    XXT = np.asmatrix(X)*np.asmatrix(X.T)
    YYT = np.asmatrix(Y)*np.asmatrix(Y.T)
    dxx = XXT.diagonal().T
    dyy = YYT.diagonal().T
    onesXX = np.asmatrix(np.ones(Y.shape[0])).T
    onesYY = np.asmatrix(np.ones(X.shape[0])).T
    A = 0.5*(dxx*onesXX.T) + 0.5*(onesYY*dyy.T) - XYT
    return gaussian(A,tauGaussian)

#Generate all Kernel matrix before experiment
def generateAllMatrices(T_d,V_d,tauGaussian) :
    for i in range(0,10) :
        K = initK(T_d[i],T_d[i],tauGaussian)
        Ktest = initK(T_d[i],V_d[i],tauGaussian)
        np.save("./Matrices/K_" + str(tauGaussian) + "_" +str(i), K)
        np.save("./Matrices/Ktest_" + str(tauGaussian) + "_" + str(i), Ktest)

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
tauGaussian = 0.1
C = 5
threshold = 10e-15

while tauGaussian < 10e4 :
    generateAllMatrices(T_d,V_d,tauGaussian)
    while C < 10e4 :
        for i in range(0,10) :
            K = np.mat(np.load("./Matrices/K_" + str(tauGaussian) + "_" +str(i) + ".npy"))
            Ktest = np.mat(np.load("./Matrices/Ktest_" + str(tauGaussian) + "_" +str(i) + ".npy"))
            print newSMO.SMO(T_d[i],T_l[i],V_d[i],V_l[i],tau,tauGaussian,C,threshold,K,Ktest)
        C = 10e4
    removeOldMatrices()
    tauGaussian = 10e4
