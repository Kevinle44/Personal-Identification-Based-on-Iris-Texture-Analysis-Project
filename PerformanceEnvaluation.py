
import numpy as np
import cv2
from sklearn.neighbors.nearest_centroid import NearestCentroid
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from sklearn.decomposition import PCA

from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching import *


def getCRRCurve(train,test):
    vec = []
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

def subClassTest(train,test):
    np.random.seed(1)
    totalClass = int(len(test)/4)
    testClass = np.random.choice(np.arange(1,totalClass+1), totalClass,replace=True)
    
    subTestSet = []
    for i in range(totalClass):
        imageIndex = np.random.choice(4,1)
        subTestSet.append(test[testClass[i]*4 + imageIndex])
    
    
    
    
    
    
    
    
