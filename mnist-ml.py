import load_mnist as loader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

(A,B)=loader.load()
print 'success'
imgplot = plt.imshow(A)
plt.show()
#np.reshape(A, A.size)
