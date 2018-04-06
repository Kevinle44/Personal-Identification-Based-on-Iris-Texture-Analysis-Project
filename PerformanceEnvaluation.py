
# Perfoemance Envaluation
# This file uses all files before and calculates the Recognition results plot for
# different dimensionality for LDA. Also, it shows the table of recognition results
# using different similarity measures. 

# In addtion, I tried to apply PCA for dimension reduction before apply LDA. The
# result was silightly increase from accuracy rate around 89% to around 90%

# Author: Zonghao Li   (zl2613@columbia.edu)
# Written in: 2018/04/04





import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching import *


# This function plots the recognition results using features of different 
# dimensionality
def getCRRCurve(train,test):
    vec = []
    # dimention could also be changed into any integer between 1 and 107, I chose
    # these as samples 
    dimention = [50,60,70,80,90,100,107]
    plt.figure()
    for i in range(len(dimention)):
        print('Currently computing dimention %d' %dimention[i])
        vec.append(getMatching(train,test,dimention[i]))
    lw = 2

    plt.plot(dimention, vec, color='darkorange',lw=lw)
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recgnition rate')
    plt.title('Recognition results using features of different dimentionality')
    plt.scatter(dimention,vec,marker='*')

    plt.show()

# Similar to getCRRCurve(), this function plots the accuracy rate for different
# dimensions for PCA. Within each PCA dimension, the maximum accuracy rate was 
# calculated by trying LDA dimensions of 90,100,107 which approves to be the dimensions
# with highest accuracy rate in general
def getPCACurve(train,test):
    train1 = train.copy()
    test1 = test.copy()
    vec = []
    pca = [400,550,600,650,1000]
    dimention = [90,100,107]
    plt.figure()
    for p in range(len(pca)):
        thisPCA = PCA(n_components=pca[p])
        thisPCA.fit(train1)
        train = thisPCA.transform(train1)
        test  = thisPCA.transform(test1)
        for i in range(len(dimention)):
            ans = []
            print('Currently computing dimention %d' %dimention[i])
            ans.append(getMatching(train,test,dimention[i]))
        vec.append(min(ans))
    lw = 2

    plt.plot(pca, vec, color='darkorange',lw=lw)
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recgnition rate')
    plt.title('Recognition results using features of different dimentionality')
    plt.scatter(pca,vec,marker='*')

    plt.show()


# This function prints the table of recognition results using different 
# similarity measures
def getTable(train,test):
    vec = []
    dimension = [100,107]
    for i in range(1,4):
        print('Currently computing distance measure number %d' %i)
        for dim in range(2):
            vec.append(getMatching(train,test,LDADimention=dimension[dim],distanceMeasure=i))
    vec = np.array(vec).reshape(3,2)
    vec = pd.DataFrame(vec)
    vec.index = ['L1 distance measure', 'L2 distance measure','Cosine similarity measure']
    vec.columns = ['Original Feature Set', 'Reduced Feature Set']
    print(vec)
    return vec


    
    
    
