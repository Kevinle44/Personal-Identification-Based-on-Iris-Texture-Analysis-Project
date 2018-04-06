
# Iris Recognition
# The main function for Iris Recognition Project
# This project implemented the Iris Recognition Algorithm Personal Identification Based on Iris Texture Analysis by Li Ma et al.
# Paper avaiable at    http://ieeexplore.ieee.org/document/1251145/
# Author: Zonghao Li   (zl2613@columbia.edu)
# Written in: 2018/04/04

#Portions of the research in this paper use the CASIA-IrisV1 collected by the Chinese Academy of Sciences' Institute of Automation (CASIA)” and a reference to “CASIA-IrisV1, http://biometrics.idealtest.org/


import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal

from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching      import *
from PerformanceEnvaluation import *


# Choose to run all the algorithm or save time runing runAllReduced()by skipping 
#image analysis part and use the pre-saved result.
def main():
    runAll()
    #runAllReduced()
    


# This function would run all the algorithm step by step including IrisLocalization,
# IrisNormalization, ImageEnhancement, Feature Extraction, IrisMatching, 
# and PerformanceEnvaluation.

# In addition to the LDA plot required by the project, I did PCA for dimension 
# reduction and ploted accuracy curve for different PCA dimensions

def runAll():
    # Run the algorithm for all training and testing images and save the result
    trainBase = getDatabase(1)
    testBase = getDatabase(2)
    irisTrain = np.array(trainBase)
    np.save('irisTrain',irisTrain)
    irisTest = np.array(testBase)
    np.save('irisTest',irisTest)
    
    # After transfering the image into vector, get performance envaluation by
    # calculating Acuracy curve for different PCA dimention reduction,
    # CRR Curve, and recognition results tables.
    train = np.load('irisTrain.npy')
    test = np.load('irisTest.npy')
    
    # Plot accuracy curve for different dimension reduction using PCA
    getPCACurve(train,test)
    
    # Plot accuracy curve for different dimensionality of the LDA
    getCRRCurve(train,test)
    
    # Draw a table for recognition results using different similarity measures
    a = getTable(train,test)



# Since the image analysis for all test and train images takes a very long time, I saved the
# result so the runAllReduced function does not run all the previous steps but directly load
# the data I saved before which saves a lot of time.

def runAllReduced():
    
    # Load train and test from data file saved before
    train = np.load('irisTrain.npy')
    test = np.load('irisTest.npy')
    # Plot accuracy curve for different dimension reduction using PCA
    getPCACurve(train,test)
    
    # Plot accuracy curve for different dimensionality of the LDA
    getCRRCurve(train,test)
    
    # Draw a table for recognition results using different similarity measures
    a = getTable(train,test)

runAllReduced()
