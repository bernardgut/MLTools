# Bernard Gutermann - 2013
##############################################################################
# mpl_plots.py
# Experiment plotter
##############################################################################
import os
import re
import numpy as np
import matplotlib.pyplot as plt

PATH = 'results/mlp'
directory = PATH+'/plots/exp1'

#plot         
def plotSingleCase(missed_train, missed_validation, error_train, error_validation, params) :
    x = range(0,30)
    missed_train = np.asarray(missed_train)
    missed_validation = np.asarray(missed_validation)
    error_train = np.asarray(error_train)
    error_validation=np.asarray(error_validation)

    plt.plot(x, missed_train[0], 'blue') 
    plt.plot(x, missed_validation[0], 'green')
    plt.legend(['Training Set', 'Validation Set'])
    plt.ylabel('Classification Mistakes')
    plt.xlabel('Epoch')
    plt.axis([0,30,0,500])
    plt.title('MLP : h1:'+str(params[0])+' eta:'+str(params[1])+' mu:'+str(params[2]))    
    filename = directory+'/M_H'+str(params[0])+'R'+str(params[1])+'M'+str(params[2])
    #plt.show()
    plt.savefig(filename+'.png')
    plt.close()
    
    G3 = plt.plot(x, error_train[0], 'blue') 
    G4 = plt.plot(x, error_validation[0], 'green')
    plt.legend(['Training Set', 'Validation Set'])
    plt.ylabel('Logistic Error')
    plt.xlabel('Epoch')
    plt.axis([0,30,0.,0.4])
    plt.title('MLP : h1:'+str(params[0])+' eta:'+str(params[1])+' mu:'+str(params[2]))
    filename = directory+'/E_H'+str(params[0])+'R'+str(params[1])+'M'+str(params[2])
    plt.savefig(filename+'.png')

#select files of interest and plot
def parsePlot(fileiter, test):

    missed_train = list()
    missed_validation = list()
    error_train = list()
    error_validation = list()

    for f in fileiter :
        match = re.search(r'results/(\w+)/h(\d+)R([0-9\.]+)M([0-9\.]+)/(\w)(\w)_(\d+).npy', f)
        tpe, h1, rho, mu, status, tset, count = match.groups()  
        #select pertinants 
        if tpe == 'mlp' and int(h1) == test[0] and float(rho) == test[1] and float(mu) == test[2] :
            if status == 'M' and tset == 'T' : 
                missed_train.append(np.load(f))
                print 'missed train : ',f
            if status == 'M' and tset == 'V' : 
                missed_validation.append(np.load(f))
                print 'missed val : ',f
            if status == 'E' and tset == 'T' : 
                error_train.append(np.load(f))
                print 'error train : ',f
            if status == 'E' and tset == 'V' : 
                error_validation.append(np.load(f))
                print 'error val : ',f
                
    plotSingleCase(missed_train, missed_validation, error_train, error_validation, test)

#Experiments
H = [2,5,15,50]
ETA = [0.1, 0.01, 0.001, 0.0001]
MU = [0.,0.2,0.4,0.6,0.8]

#recover all files
fileiter = (os.path.join(root, f)
    for root, _, files in os.walk(PATH)
    for f in files)

Files = list()
for f in fileiter :
    Files.append(f)
    
if not os.path.exists(directory):
        os.makedirs(directory)
#save plots
for h1 in range(0,len(H)) :
    for eta in range(0,len(ETA)) : 
        for mu in range(0,len(MU)) :
            print 'saving plot : h1',H[h1],'R', ETA[eta],'MU', MU[mu]
            test = [H[h1], ETA[eta], MU[mu]]
            parsePlot(Files, test)
#test = [H[0], ETA[0], MU[0]]
#parsePlot(fileiter, test)
