# Bernard Gutermann - 2013
###########################################################################
# mlp_experiments.py
# Experiences on the mlp for various parameters
#
# output files are saved in the output/mlp folder
########################################################################### 

import mlp
import numpy as np
import os

#Experiments
H = [2,10,40,50,60]
ETA = [0.07,0.05,0.03, 0.01, 0.005]
MU = [0,0.1,0.2,0.4,0.6,0.7]

PATH = 'results/mlp/test'
if not os.path.exists(PATH):
    os.makedirs(PATH)

def multiRunOptimal():
    X=np.load('mnist/n_MNIST_Test49.npy')
    L=np.load('mnist/n_MNIST_Test_labels49.npy')
    
    Error = list()
    Mistakes = list()
    
    e = 0
    for i in range(0,20) :
        print 'Begin iteration ',i
        W = mlp.run(50,0.05,0.2,i)
        (E, M) = mlp.test(X,L,W)
        
        Error.append(E)
        Mistakes.append(M)
        
        e=e+1
        print 'test error :\t',E
        print 'test mistakes :\t',M
    
    #save
    print 'Saving Error trace : ', Error
    print 'Saving Mistakes trace : ',Mistakes
    np.save(PATH+'/result_E5',np.asarray(Error))
    np.save(PATH+'/result_M5',np.asarray(Mistakes)) 

def affineSearch():
    e = 0
    for h1 in range(0,len(H)) :
        for eta in range(0,len(ETA)) : 
            for mu in range(0,len(MU)) :
                print '###BEGIN EXPERIMENT : ',e,' : h1=',H[h1],' eta=',ETA[eta],' mu=',MU[mu]
                mlp.run(H[h1],ETA[eta],MU[mu],e)
                print '###END EXPERIMENT : ',e,' : h1=',H[h1],' eta=',ETA[eta],' mu=',MU[mu]
                e=e+1
                
#affineSearch()
multiRunOptimal()
