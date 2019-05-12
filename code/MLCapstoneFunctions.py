# -*- coding: utf-8 -*-
"""
Created on Sun Dec 3 10:00:00 2017
Last modified on Fri Dec 22 9:25:00 2017 

@author: Alfonso Sanchez De Lucio
"""
'''****************************************************************************
Importing libraries
****************************************************************************'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import nitime as ni
import nitime.algorithms as tsa 
from sklearn.decomposition import PCA
'''****************************************************************************
Rreading Data function
****************************************************************************'''
#Function to read the data 
def reading_data(fname,goal):
    
    """Function to get data containing in the training and tested folders. The
    data that will be obtained are the EEG datasets and the labels that repres-
    ents the events of interest.
    
    -Inputs:
        fname  -> Name of the file where the data is going to be read
        goal   -> Variable to determine if the data is going to be used for tr-
                  aining or for testing.
    -Returns:
        data   -> The EEG data read from the file
        labels -> The labels realted with the data that was read."""
        
    #Reading of the EEG data
    data = pd.read_csv(fname)
    events_fname = fname.replace('_data','_events')
    labels= pd.read_csv(events_fname)

    if goal=="training":
        data=data.drop(['id' ], axis=1)#remove id
        labels=labels.drop(['id' ], axis=1)#remove id
    elif goal=="testing":
        labels=labels.drop(['id' ], axis=1)
    else:
        raise SystemExit("The goal variable is unknown for the function")

    return data, labels

'''****************************************************************************
Obtaining features for a SVM classifier function
****************************************************************************'''
#Funtion to preprocess the EEG data and their labels
def data_preprocessing_PSD(X, y, ids, funct='train',FramesLen=150, ARorder=6, Fs=500, Fo=0,Fr=250):

    """Function to preprocess the EEG data and to obtain the power spectral de-
    nsities of the EEG channels to be used as feautures for the classifiers. 
    The PSDs are estimated based on the EEG signals AR coefficients.
    
    -Inputs:
        X  -> Data to train or test the classifier
        y   -> Labels for the data used for the training or testing process
        ids   -> Ids for the dataset subjects
        funct -> Varibale used to let the algorithm know if the preprocess data
                 is going to being used  for the training or testing of a cla-
                 ssifier
        FramesLen -> Length of the sements in which will be partionated the 
                     EEG channels data.
        ARorder -> Order for the modelling of the EEG segmented signals as
                   Autoregressive processes.
        Fs -> Sampling frequency for the EEG data
        Fo -> Smaller frequency to acquire from the PSDs
        Fr -> Maximun frequency to acquire from the PSDs
        
    -Returns:
        Output:
               X_test_temp -> The preprocesses segmented EEG signals, in the form
                              of their PSD features.
               y_test_temp -> The preprocessed segmenetd EEG labels
               indices -> The indices for each of row of the segment EEG data """
        
    #substrating ghe mean from the data
    X_prep=StandardScaler().fit_transform(X)
    
    #Defining the indices that separates the EEG into different segmetns
    FrameInit=[]
    FrameLimit=[]
    indices = []
    Frame=0
        
    #Finding the indices that separates the EEG data into different segments
    while (Frame)<=X_prep.shape[0]:
        if (Frame+FramesLen)>X_prep.shape[0]:
            FrameLimit.append(X_prep.shape[0])
            FrameInit.append(X_prep.shape[0]-FramesLen)
        else:
            FrameLimit.append(Frame+FramesLen)    
            FrameInit.append(Frame)  
        Frame+=FramesLen
        
    #Creating the numpy array space to store each segment of the EEG data
    #Space created to store Fr frequencies, features, for each EEG channel      
    X_test_temp=np.zeros(shape=(len(FrameInit),(Fr-Fo)*X_prep.shape[1]),dtype=float)
    y_test_temp=np.zeros(shape=(len(FrameInit),y.shape[1]),dtype=float)           
    
    #Partitioning the EEG data 
    Index=0
    for FrameO,FrameL in zip(FrameInit,FrameLimit):
        for channel in range(X.shape[1]):     
            #Genereting the PSDs of the segements EEG data based on their AR coefficients
            ARpar=ni.algorithms.autoregressive.AR_est_YW(X_prep[FrameO:FrameL,channel], ARorder, rxx=None)
            alpha=ARpar[0]
            sigma_v=ARpar[1]
            ARpsd = ni.algorithms.autoregressive.AR_psd(alpha, sigma_v, n_freqs=Fs, sides='onesided')
            AR_psd=ARpsd[1]
            X_test_temp[Index,channel*(Fr-Fo):(channel+1)*(Fr-Fo)]=AR_psd[Fo:Fr]
                
        #Preprocessing the labels in order to segment the EEG data
        for action in range(y.shape[1]):
            if sum(y[FrameO:FrameL,action])>0:
                y_test_temp[Index,action]=1

        #Defining if the data will ne used for training or testing a classifier
        if funct=='test':
            #Creating indices to generate dataframes with the results obtained testing the classifiers
            tempindex=ids[FrameO+1][8:].find('_')
            indices.append(ids[FrameO+1][0:8+tempindex+1]+str(FrameO)+'-'+str(FrameL))               
        elif funct!='train':
            raise SystemExit("The specified function is unknown")
                
        Index+=1
 
    #Defining the output of the function
    if funct=='train':
        Output=[X_test_temp,y_test_temp]
    else:
        Output=[X_test_temp,  y_test_temp, indices]
        
    return Output

'''****************************************************************************
Obtaining features for an MLP classifier function
****************************************************************************'''
#Funtion to preprocess the EEG data and their labels
def data_preprocessing_AR(X, y, ids, funct='train',FramesLen=150, ARorder=6):

    """Function to preprocess the EEG data and to obtain the AR coefficients
    of the segmented signals to be used as feutures to train and test 
    BCI classifiers
    
    -Inputs:
        X  -> Data to train or test the classifier
        y   -> Labels for the data used for the training or testing process
        ids   -> Ids for the dataset subjects
        funct -> Varibale used to let the algorithm know if the preprocess data
                 is going to being used  for the training or testing of a cla-
                 ssifier
        FramesLen -> Length of the sements in which will be partionated the 
                     EEG channels data.
        ARorder -> Order for the modelling of the EEG segmented signals as
                   Autoregressive processes.
        
    -Returns:
        Output:
               X_test_temp -> The preprocesses segmented EEG signals, in the form
                              of their PSD features.
               y_test_temp -> The preprocessed segmenetd EEG labels
               indices -> The indices for each of row of the segment EEG data """
        
    #Subtratinc ghe mean from the data
    X_prep=StandardScaler().fit_transform(X)
    
    #Defining the indices that separates the EEG into different segmetns
    FrameInit=[]
    FrameLimit=[]
    indices = []
    Frame=0
        
    #Finding the indices that separates the EEG data into different segments
    while (Frame)<=X_prep.shape[0]:
        if (Frame+FramesLen)>X_prep.shape[0]:
            FrameLimit.append(X_prep.shape[0])
            FrameInit.append(X_prep.shape[0]-FramesLen)
        else:
            FrameLimit.append(Frame+FramesLen)    
            FrameInit.append(Frame)  
        Frame+=FramesLen
        
    #Creating the numpy array space to store each segment of the EEG data
    #Space created to store AR * number of EEG channels as features
    X_test_temp=np.zeros(shape=(len(FrameInit),(X_prep.shape[1])*ARorder),dtype=float)
    y_test_temp=np.zeros(shape=(len(FrameInit),y.shape[1]),dtype=float)       

    #Partitioning the EEG data 
    Index=0
    for FrameO,FrameL in zip(FrameInit,FrameLimit):
        for channel in range(X.shape[1]):     
            #Obtained the AR coefficients for the segmentes EEG signals
            ARfeatures=ni.algorithms.autoregressive.AR_est_YW(X_prep[FrameO:FrameL,channel], ARorder, rxx=None)[0]
            X_test_temp[Index,(channel)*ARorder:(channel+1)*ARorder]=ARfeatures
                
        #Preprocessing the labels in order to segment the EEG data
        for action in range(y.shape[1]):
            if sum(y[FrameO:FrameL,action])>0:
                y_test_temp[Index,action]=1

        #Defining if the data will ne used for training or testing a classifier
        if funct=='test':
            #Creating indices to generate dataframes with the results obtained testing the classifiers
            tempindex=ids[FrameO+1][8:].find('_')
            indices.append(ids[FrameO+1][0:8+tempindex+1]+str(FrameO)+'-'+str(FrameL))                
        elif funct!='train':
            raise SystemExit("The specified function is unknown")
                
        Index+=1
 
    #Defining the output of the function
    if funct=='train':
        Output=[X_test_temp,y_test_temp]
    else:
        Output=[X_test_temp,  y_test_temp, indices]
        
    return Output

'''****************************************************************************
Obtaining features for an MLP classifier function
****************************************************************************'''

def data_preprocessing_TA(X):
    
    """Function that preprocess the data used to train and test the classifiers
    
    -Inputs:
        X  -> Data to train or test the classifier
        
    -Returns:
        X_prep  -> Preprocessed data"""
    
    #Removing the mean and scaling the data
    X_prep=StandardScaler().fit_transform(X)
    #do here your preprocessing
    return X_prep

'''****************************************************************************
Function to reduce the features to consider, PCA
****************************************************************************'''

#    pca = PCA(n_components=10,whiten=True)
#    X_train = pca.fit_transform(X_train[:,channels])
#    X_test = pca.transform(X_test[:,channels])
    
'''****************************************************************************
Fin
****************************************************************************'''