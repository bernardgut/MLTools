import numpy as np
import matplotlib.pyplot as plt

X = np.mat(np.load("../mnist/n_MNIST_Test49.npy"))
error = X[138,:].reshape((28, 28))
plt.imshow(error)
plt.show()
