import numpy as np
import newSMO
import KernelComputation

#initialization
tau = 10e-8
#tauGaussian = 0.1
#C = 5
threshold = 10e-15
#Loading data
Xtest = np.mat(np.load("../mnist/n_MNIST_Test49.npy"))
Ttest = np.mat(np.load("../mnist/n_MNIST_Test_labels49.npy"))
X = np.mat(np.load("../mnist/n_MNIST_Complete49.npy"))
T = np.mat(np.load("../mnist/n_MNIST_Complete49_Label.npy"))

#Info about shapes
print Ttest.shape
print Xtest.shape
print X.shape
print T.shape

K = KernelComputation.initK(X,tauGaussian=0.09)
print "K generated"
Ktest = KernelComputation.initK(X,Xtest,tauGaussian=0.09)
print "Ktest generated"
Alpha = newSMO.SMO(X,T,tau,0.09,3.,threshold,K,5000)
print Alpha.shape
pred = newSMO.prediction(Alpha,3.,T,Ktest, K)
print "Final number of error : ",newSMO.getError(pred,Ttest)
missclassified = newSMO.getIndexMissclassified(pred,Ttest)
print "Index of class 1 ",missclassified[0]
print "index of class 2 ",missclassified[1]
