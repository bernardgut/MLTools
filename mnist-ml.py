import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#load data
T=np.load('mnist/nTrainingSet.npy')
V=np.load('mnist/nValidationSet.npy')
print 'load success'
print T.shape,' ',V.shape

#display data
#imgplot = plt.imshow(A)
#plt.show()
#np.reshape(A, A.size)




