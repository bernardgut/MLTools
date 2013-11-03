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

#min max
max = np.amax(A)
min = np.amin(A)
#normalize
J = min*np.ones(A.shape)
N = (A-J)/(max-min)

#split data
(T,V)=splitter.split(N)
print 'Training.training : ', T.shape,'; Training.validation : ',V.shape, ' ; writing to disk...'
np.save('mnist/nTrainingSet',T)
np.save('mnist/nValidationSet',V)


