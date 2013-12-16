import numpy as np
import matplotlib.pyplot as plt

X = np.mat(np.load("../mnist/n_MNIST_Test49.npy"))
L = np.mat(np.load("../mnist/n_MNIST_Test_labels49.npy"))

print "Label : ",L[1482,:]
error = X[1482,:].reshape((28, 28))
plt.imshow((np.mat(error)).T)
plt.ylabel('')
plt.xlabel('')
plt.show()
