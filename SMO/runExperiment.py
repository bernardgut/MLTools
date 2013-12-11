import numpy as np

#Initialize the kernel matrix 
def initK(X,Y, tauGaussian) :
    print X.shape, Y.shape
    XYT = np.asmatrix(X)*np.asmatrix(Y.T)
    XXT = np.asmatrix(X)*np.asmatrix(X.T)
    YYT = np.asmatrix(Y)*np.asmatrix(Y.T)
    dxx = XXT.diagonal().T
    dyy = YYT.diagonal().T    
    onesXX = np.asmatrix(np.ones(Y.shape[0])).T
    onesYY = np.asmatrix(np.ones(X.shape[0])).T
    A = 0.5*(dxx*onesXX.T) + 0.5*(onesYY*dyy.T) - XYT
    return gaussian(A,tauGaussian)

#Gaussian function
def gaussian(A,tauGaussian) :
    return np.exp(-tauGaussian*A)

#Generate all Kernel matrix before experiment
def generateAllKmatrices(A,L,tauGaussian) :
    for i in range (0,10) :
        X,T,V,V_l = split10Fold.split(A,L,i)
        K = initK(X,X,tauGaussian)
        Ktest = initK(X,V,tauGaussian)
        np.save("./KernelMatrices/K_" + str(tauGaussian) + "_" +str(i), K)
        np.save('./KernelMatrices/Ktest_' + str(tauGaussian) + "_" + str(i), Ktest)
    return (allK,allKtest)

#Load data
A=np.load("./Data_Label/A.npy")
L=np.load("./Data_Label/L.npy")
#Load 10-fold datasets
T_d = list()
T_l = list()
V_d = list()
V_l = list()
for i in range(0,10) :
    T_d.append(np.load("./Data_Label/T_d_"+ str(i) + ".npy"))
    T_l.append(np.load("./Data_Label/T_l_"+ str(i) + ".npy"))
    V_d.append(np.load("./Data_Label/V_d_"+ str(i) + ".npy"))
    V_l.append(np.load("./Data_Label/V_l_"+ str(i) + ".npy"))
#Parameters
tau = 10e-8
tauGaussian = 0.1
C = 5
threshold = 10e-15
#little xor problem
#X = np.matrix('(-1,-1);(1,-1);(-1,1);(1,1)')
#V = np.matrix('(-1,-1);(1,-1);(-1,1);(1,1)')
#T = np.matrix('(-1);(1);(1);(-1)')
#V_l = np.matrix('(-1);(1);(1);(-1)')
#(A,B,C,D) = split(X,T,3)

f = open('result.txt','w')
f.write("(Misclassified,C,tauGaussian)\n")
#Loop over C to seek the best one
while tauGaussian < 2e5 :
    #Loop over tau to seek the best one
    generateAllKmatrices(A,L,tauGaussian)
    while C < 2e5 :     
        totalNotCorrect = 0
        #10-fold validation
        for i in range(0,10) :
            K = np.load("./KernelMatrices/K_" + str(tauGaussian) + "_" +str(i) + ".npy")
            Ktest = np.load("./KernelMatrices/Ktest_" + str(tauGaussian) + "_" +str(i) + ".npy")
            print i,"eme fold", "tauGaussian", tauGaussian, "C",C
            totalNotCorrect = totalNotCorrect + SMO(T_d[i],T_l[i],V_d[i],V_l[i],tau,tauGaussian,C,allK[i],allKtest[i])
        totalNotCorrect = totalNotCorrect / 10
        message = totalNotCorrect,C,tauGaussian
        currentresult = str(message)
        f.write(currentresult)
        f.write("\n")
        tauGaussian = tauGaussian * 10
        print "=================="
        C = 2e5
    tauGaussian = 2e5
    print "**********"
f.close()
