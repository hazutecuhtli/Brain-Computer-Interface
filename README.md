
# Use of the script

The file MLCapstoneProject.ipynb has the main code neccesary to run this project. The use of the script is very simple, just some small variables needs to be changed to get different results. 

The algoritm was set up to run the proposed project solution, however, if required,the bencmark classifier can be tested by commenting the following line:

## Selecting the classifier

'''In case a SVC classifier is selected'''

---> clf = SVC(kernel='poly',C=1, gamma=1, degree=3, probability=True, random_state=0) <--- 

Then, by uncommenting the following line:

'''In case a MLP classifier is selected'''

## Important

The folder containing the file MLCapstoneProject.ipynb needs to have two folders that are used to contains the training and testing dataset. The folder containing the training data needs to be called "train", while the folder containing the testing data needs to be named "test". If other data wants to be use, the files located in both folders can be replaced. However, the format of the filenames need to be the same. 

## Functions

Also, the main algorithm needs the MLCapstoneFunctions.py and MLCapstoneVisuals.py  files in order to works properly. These two files need to be located in the same folder as the MLCapstoneProject.ipynb script. 

## Datasets

The dataset used can be obtained from the data competition in the following link:

https://www.kaggle.com/c/grasp-and-lift-eeg-detection/data

However, only the file train.zip is used. The obtained data from the kaggle competition needs to be divided, for the training and testing datasets. For this, 1 to 6 series of data and events, for all 12 subjects, need to be located in the "train" folder, while the remaining series of data and events need to be located in the "test" folder. This is the setup required in order to get the results presented in this project. 

Additionally, the folder with the data used in this project can also be downloaded from the following link:

https://www.dropbox.com/s/nszkyfvvtg5paa4/CapstoneProject.zip?dl=0

If the data is obtained from the last link, the test and train folders, in the project folder, need to be replaced.
