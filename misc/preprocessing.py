#Normalizer module
import load_mnist as loader
import splitter as splitter

import numpy as np

#load data
(A,L)=loader.loadTrainingSet()
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
(T, T_l, V, V_l)=splitter.split(N, L)
print T.shape
print 'Training.training : ', T.shape, ' l : ',T_l.shape, ' ; Training.validation : ',V.shape, ' l : ', V_l.shape, '\n ; writing to disk...'

np.save('../mnist/nTrainingSet',T)
np.save('../mnist/nValidationSet',V)
np.save('../mnist/nTrainingSet_labels',T_l)
np.save('../mnist/nValidationSet_labels',V_l)
