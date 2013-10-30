import load_mnist as loader
import splitter as splitter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

(A,B)=loader.loadTrainingSet()
print 'success'
#imgplot = plt.imshow(A)
#plt.show()
#np.reshape(A, A.size)
(T,V)=splitter.split(A)
print T.shape,' ',V.shape
