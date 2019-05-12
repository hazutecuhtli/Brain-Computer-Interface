# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:36:21 2017
Last modified on Fri Dec 22 10:28:00 2017 

@author: Alfonso Sanchez De Lucio
"""
'''****************************************************************************
Importing libraries
****************************************************************************'''
from sklearn.metrics import classification_report
import matplotlib.pylab as plt

'''****************************************************************************
Plotting ROC curves function
****************************************************************************'''

def Plotting_ROCcurve(events, fpr, tpr, roc_auc):

    """Function to plot the ROC curves and the area under the ROC curves
    
    -Inputs:
        events  -> Name of the events of interst to classify
        fpr   -> False positive rate
        tpr -> True positive rate
        roc_auc -> ROC area under the curves """

    #Plotting
    SMALL_SIZE = 10
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 12

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('xtick', labelsize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('ytick', labelsize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('legend', fontsize=SMALL_SIZE)          # controls default text sizes
    
    plt.figure()
    lw = 2
    for k in range(len(events)):
        plt.plot(fpr[k], tpr[k], lw=lw, label='ROC curve ' +  events[k] + ' (area = %0.2f)' % roc_auc[k])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curves')
    plt.legend(loc="lower right")

    plt.show()
   
'''****************************************************************************
Plotting relationships between features 
****************************************************************************'''

def FeaturesRelationsLoop(Relationships, Features):
    
    """Function to plot the correlations between the classifiers features
    
    -Inputs:
            Relationships: Features correlations
            Features: Name of the feautures used, string."""
    
    #Plotting
    SMALL_SIZE = 17
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title

    for n in range(4):
        for m in range(3):    
            plt.subplots()
            axes = plt.gca()
            tempvar='subject' + str((m+1)+(n*3))
            axes.set_xlim([0,Relationships[tempvar].shape[0]-1])
            axes.set_ylim([0,Relationships[tempvar].shape[0]-1])
            plt.grid()
            plt.ylabel('EEG ' + Features + ' features')
            plt.xlabel('EEG ' + Features + ' features')
            plt.title('Subject ' + str((m+1)+(n*3)) + ' features correlation')
            gra = plt.imshow(abs(Relationships[tempvar]), cmap='jet')
            cbar = plt.colorbar(gra, ticks=[0, 1], orientation='vertical')
            plt.show()
            
'''****************************************************************************
Printing matrix of results 
****************************************************************************'''

def AverageResults(Accuracy, Precision, events, features):
    
    """Funtion to print obtained results
    
    -Inputs:
            Acuracy: Average of the accuracy obtained for each movement of interest
            Precision: Average of the precision obtained for each movement of interest
            events: Name of the events of interst, string.
            features: Name of the features used to obtain the results, string."""
    
   #Printing
   print()
   print("************************   BCI Average " + features + " Capstpne Project results    ***************************\n")
   
   rows = 3
   cols = 6
   A=events.copy()
   A.append('Metric\Event')
   Spaces=[8,13,8,3,5,5]
   for n in range(0,rows):
       for i in range(0,cols+1):
           if n==0:
               print(A[i-1] + '    ', end="")
           if n==1:
               if i==0:
                   print('  Accuracy       ', end="")
               else:
                   B=' '
                   C=B.zfill(Spaces[i-1])
                   print("%6.3f" %Accuracy[i-1], C.replace('0',' ')  , end=" ")
           if n==2:
               if i==0:
                   print(' Precision       ', end="")
               else:
                   B=' '
                   C=B.zfill(Spaces[i-1])
                   print("%6.3f" %Precision[i-1], C.replace('0',' ')  , end=" ")                   
       print() 
   print()    
    
               
'''****************************************************************************
Fin
****************************************************************************'''