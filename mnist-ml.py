import load_mnist as loader
import splitter as splitter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#load data
(A,B)=loader.loadTrainingSet()
print 'success'

#display data
#imgplot = plt.imshow(A)
#plt.show()
#np.reshape(A, A.size)

#split data
(T,V)=splitter.split(A)
print T.shape,' ',V.shape

x = [[1, 2, 10 ,1 ,2],[ 1,2,12,0,10]]
print np.amax(x)
