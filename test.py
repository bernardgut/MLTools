import numpy as np
import random

A = np.array([[1,2,3],[4,5,6],[7,8,9]])

run = 0
i = 0.001
j = 0.2
filename = 'r'+str(run)+'-'+str(i)+'-'+str(j) 
np.save(filename, A)
