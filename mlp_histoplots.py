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
directory = 'results/mlp/plots/bar_plots'
#plot         
def plotHist(error_train, error_validation) :
    
    x = np.asarray(range(0,len(error_train)))

    error_train = np.asarray(error_train)
    error_validation=np.asarray(error_validation)
    
    print params_ET
    #plot the histogramm
    plt.bar(x, error_train)
    plt.ylabel('Normalized Logistic Error')
    plt.xlabel('Parameters')
    plt.title('MLP : Error Distribution Sample for [h1, eta, mu]')  
    plt.xticks(x+0.5,params_ET, rotation='vertical')
    filename = directory+'/error_train_hist'
    plt.show()
    #plt.savefig(filename+'.pdf')
    #plt.close()
    
    print params_EV
    #plot the histogramm
    plt.bar(x, error_validation)
    plt.ylabel('Normalized Logistic Error')
    plt.xlabel('Parameters')
    plt.title('MLP : Error Distribution Sample for [h1, eta, mu]')   
    plt.xticks(x+0.5,params_EV, rotation='vertical')
    filename = directory+'/error_val_hist'
    plt.show()
    #plt.savefig(filename+'.pdf')
    #plt.close()
    """
    #plt.legend(params_MT)
    plt.ylabel('Classification Mistakes [%]')
    plt.xlabel('Epoch')
    plt.axis([0,30,0,500.*100./2000.])
    plt.title('MLP : Training Classification Mistakes - for [h1, eta, mu]')    
    filename = directory+'/misstakes_validation'
    #plt.show()
    plt.savefig(filename+'.pdf')
    plt.close()
    """
#select files of interest and plot
def parsePlot(fileiter, test):

    for f in fileiter :
        match = re.search(r'/mlp/h(\d+)R([0-9\.]+)M([0-9\.]+)/(\w)(\w)_(\d+).npy', f)
        if match!=None :
            h1, rho, mu, status, tset, count = match.groups()  
            #select pertinants 
            if int(h1) == test[0] and float(rho) == test[1] and float(mu) == test[2] :
                if status == 'E' and tset == 'T' : 
                    error_train.append(np.min(np.load(f)))
                    params_ET.append(str(test))
                    print 'error train : ',f
                if status == 'E' and tset == 'V' : 
                    error_validation.append(np.min(np.load(f)))
                    params_EV.append(str(test))
                    print 'error val : ',f
                    

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
        
error_train = list()
error_validation = list()
params_ET = list()
params_EV = list()

#save plots
for h1 in range(0,len(H)) :
    for eta in range(0,len(ETA)) : 
        for mu in range(0,len(MU)) :
            print 'saving plot : h1',H[h1],'R', ETA[eta],'MU', MU[mu]
            test = [H[h1], ETA[eta], MU[mu]]
            parsePlot(Files, test)

plotHist(error_train, error_validation)
#test = [H[0], ETA[0], MU[0]]
#parsePlot(fileiter, test)
