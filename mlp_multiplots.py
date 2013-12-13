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
directory = 'results/mlp/plots/multi_plots'

#plot         
def plotSingleCase(missed_train, missed_validation, error_train, error_validation) :
    
    x = range(0,30)
    missed_train = np.asarray(missed_train)
    missed_validation = np.asarray(missed_validation)
    error_train = np.asarray(error_train)
    error_validation=np.asarray(error_validation)
    
    #plot the 4 graphs
    for i in range(0,len(missed_train)) :
        plt.plot(x, missed_train[i]*100./2000.) 
        plt.legend(params_MT)
    plt.ylabel('Classification Mistakes [%]')
    plt.xlabel('Epoch')
    plt.axis([0,30,0,500.*100./2000.])
    plt.title('MLP : Training Classification Mistakes - for [h1, eta, mu]')    
    filename = directory+'/misstakes_validation'
    #plt.show()
    plt.savefig(filename+'.pdf')
    plt.close()
    
    for i in range(0,len(missed_validation)) :
        plt.plot(x, missed_validation[i]*100./2000.) 
        plt.legend(params_MT)
    plt.ylabel('Classification Mistakes [%]')
    plt.xlabel('Epoch')
    plt.axis([0,30,0,500.*100./2000.])
    plt.title('MLP : Validation Classification Mistakes - for [h1, eta, mu]')    
    filename = directory+'/misstakes_train'
    #plt.show()
    plt.savefig(filename+'.pdf')
    plt.close()
    
    for i in range(0,len(error_train)) :
        plt.plot(x, error_train[i]) 
        plt.legend(params_MT)
    plt.ylabel('Normalized Logistic Error')
    plt.xlabel('Epoch')
    plt.axis([0,30,0,0.5])
    plt.title('MLP : Training Classification Error - for [h1, eta, mu]')    
    filename = directory+'/error_train'
    #plt.show()
    plt.savefig(filename+'.pdf')
    plt.close()
    
    for i in range(0,len(error_train)) :
        plt.plot(x, error_validation[i]) 
        plt.legend(params_MT)
    plt.ylabel('Normalized Logistic Error')
    plt.xlabel('Epoch')
    plt.axis([0,30,0,0.5])
    plt.title('MLP : Validation Classification Error - for [h1, eta, mu]')    
    filename = directory+'/error_validation'
    #plt.show()
    plt.savefig(filename+'.pdf')
    plt.close()
    
    """
    G3 = plt.plot(x, error_train[0], 'blue') 
    G4 = plt.plot(x, error_validation[0], 'green')
    plt.legend(['Training Set', 'Validation Set'])
    plt.ylabel('Logistic Error')
    plt.xlabel('Epoch')
    plt.axis([0,30,0.,0.4])
    plt.title('MLP : h1:'+str(params[0])+' eta:'+str(params[1])+' mu:'+str(params[2]))
    filename = directory+'/E_H'+str(params[0])+'R'+str(params[1])+'M'+str(params[2])
    plt.savefig(filename+'.png')
    """
#select files of interest and plot
def parsePlot(fileiter, test):

    for f in fileiter :
        match = re.search(r'/mlp/h(\d+)R([0-9\.]+)M([0-9\.]+)/(\w)(\w)_(\d+).npy', f)
        
        #select pertinants 
        if match!=None :
            h1, rho, mu, status, tset, count = match.groups()  
            if int(h1) == test[0] and float(rho) == test[1] and float(mu) == test[2] :
                if status == 'M' and tset == 'T' : 
                    missed_train.append(np.load(f))
                    params_MT.append(test)
                    print 'missed train : ',f
                if status == 'M' and tset == 'V' : 
                    missed_validation.append(np.load(f))
                    params_MV.append(test)
                    print 'missed val : ',f
                if status == 'E' and tset == 'T' : 
                    error_train.append(np.load(f))
                    params_ET.append(test)
                    print 'error train : ',f
                if status == 'E' and tset == 'V' : 
                    error_validation.append(np.load(f))
                    params_EV.append(test)
                    print 'error val : ',f
                    

#Experiments
H = [40,50,60]
ETA = [0.05, 0.01, 0.005]
MU = [0,0.1,0.2]

#recover all files
fileiter = (os.path.join(root, f)
    for root, _, files in os.walk(PATH)
    for f in files)

Files = list()
for f in fileiter :
    Files.append(f)
    
if not os.path.exists(directory):
        os.makedirs(directory)
        
missed_train = list()
missed_validation = list()
error_train = list()
error_validation = list()
params_MT = list()
params_MV = list()
params_ET = list()
params_EV = list()

#save plots
for h1 in range(0,len(H)) :
    for eta in range(0,len(ETA)) : 
        for mu in range(0,len(MU)) :
            print 'saving plot : h1',H[h1],'R', ETA[eta],'MU', MU[mu]
            test = [H[h1], ETA[eta], MU[mu]]
            parsePlot(Files, test)

plotSingleCase(missed_train, missed_validation, error_train, error_validation)
#test = [H[0], ETA[0], MU[0]]
#parsePlot(fileiter, test)
