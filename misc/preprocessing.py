#Normalizer module
import load_mnist as loader
import splitter as splitter

import numpy as np

#load data
(A,L)=loader.loadTrainingSet()
print 'success'
print A.shape
#display data
#imgplot = plt.imshow(A)
#plt.show()
#np.reshape(A, A.size)

#min max
amax = np.amax(A)
amin = np.amin(A)
#normalize
J = amin*np.ones(A.shape)
N = (A-J)/(amax-amin)

#split data
(T, T_l, V, V_l)=splitter.split(N, L)
print 'Training.training : ', T.shape, ' l : ',T_l.shape, ' ; Training.validation : ',V.shape, ' l : ', V_l.shape, '\n ; writing to disk...'

np.save('../mnist/n_MNIST_Training',T)
np.save('../mnist/n_MNIST_Validation',V)
np.save('../mnist/n_MNIST_Training_labels',T_l)
np.save('../mnist/n_MNIST_Validation_labels',V_l)
