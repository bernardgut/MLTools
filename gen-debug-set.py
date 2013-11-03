#Normalizer module
import numpy as np
from scipy.misc import imresize
T = np.load("mnist/nTrainingSet.npy")
y = np.empty((50,144))
for i in range(0,50) :
	x = T[i].reshape(28,28)
	newimg = imresize(x, (12,12),'nearest')
	y[i] = newimg.reshape(1,144)
print "reshape finished"

np.save('debugTestSet',y)




