import numpy as np
import newSMO
import KernelComputation

#Load the 10-fold dataset
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
threshold = 10e-15
tauSetPrecise = [0.05,0.07,0.09,0.1,0.11,0.12,0.15]
cSetPrecise = [1,3,5,7,9,10]
#tauSet = [0.001]
#cSet = [0.1,1.0,10.0,100.0]
#tauGaussian = 10e-3
#C = 10e-2

for tauGaussian in tauSetPrecise :
    KernelComputation.generateAllMatrices(T_d,V_d,tauGaussian)
    for C in cSetPrecise :
        nbError = 0
        for i in range(0,10) :
            K = np.mat(np.load("./Matrices/K_" + str(tauGaussian) + "_" +str(i) + ".npy"))
            Ktest = np.mat(np.load("./Matrices/Ktest_" + str(tauGaussian) + "_" +str(i) + ".npy"))
            nbError = nbError + newSMO.SMO(np.mat(T_d[i]),np.mat(T_l[i]),np.mat(V_d[i]),np.mat(V_l[i]),tau,tauGaussian,C,threshold,K,Ktest,500)
            print "i,tauGaussian,C,error",i,tauGaussian,C,nbError
        nbError = nbError / 10.0
        print "average error : ", nbError
        f = open("./ResultPrecis2/result_" + str(tauGaussian) + "_" +str(C),"w")
        f.write(str(nbError))
        f.close()
    KernelComputation.removeOldMatrices()
