# Bernard Gutermann - 2013
###########################################################################
# mlp_experiments.py
# Experiences on the mlp for various parameters
#
# output files are saved in the output/mlp folder
########################################################################### 

import mlp

#Experiments
H = [2,5,15,50]
ETA = [0.1, 0.01, 0.001, 0.0001]
MU = [0,0.2,0.4,0.6,0.8]

e = 0

for h1 in range(0,len(H)) :
    for eta in range(0,len(ETA)) : 
        for mu in range(0,len(MU)) :
            print '###BEGIN EXPERIMENT : ',e,' : h1=',H[h1],' eta=',ETA[eta],' mu=',MU[mu]
            mlp.run(H[h1],ETA[eta],MU[mu])
            print '###END EXPERIMENT : ',e,' : h1=',H[h1],' eta=',ETA[eta],' mu=',MU[mu]
            e=e+1
