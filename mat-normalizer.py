#Normalizer module
import load_mnist as loader
import splitter as splitter

import numpy as np

#load data
(A,LABELS)=loader.loadTrainingSet()
print 'success'

#display data
#imgplot = plt.imshow(A)
#plt.show()
#np.reshape(A, A.size)

#split data
(T,V)=splitter.split(A)
print 'Training.training : ', T.shape,'; Training.validation : ',V.shape
#min max
max = np.amax(T)
min = np.amin(T)
#normalize
J = min*np.ones(T.shape)
N = (T-J)/(max-min)


