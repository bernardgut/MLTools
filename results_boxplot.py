import numpy as np
import pylab

PATH = 'results/mlp/test'
OPATH = PATH+'/plots/final'

E1 = np.load(PATH+'/result_E1.npy')
E2 = np.load(PATH+'/result_E.npy')
pylab.figure()
pylab.boxplot([E1,E2])

M1 = np.load(PATH+'/result_M1.npy')
M2 = np.load(PATH+'/result_M.npy')
pylab.figure()
pylab.boxplot([M1/1902.,M2/1902.])
pylab.show()
