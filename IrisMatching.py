
# Iris Matching
# This file uses all files before to get the training and testing database from
# the raw images. After we transfer all image into vector, do LDA and matching to
# get the accuracy rate

# Author: Zonghao Li   (zl2613@columbia.edu)
# Written in: 2018/04/04





import numpy as np
import cv2
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *


# This function uses all modules before. For each fileName, first read in the file
# as image, then do Iris Localization, Normalization, Image Enhancement, and then
# extract features from that.
def processImage(fileName):
    img = cv2.imread(fileName)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (innerCircle,outterCircle) = irisLocalization(image)
    image = getNormalization(image,innerCircle,outterCircle)
    image = allEnhancement(image)
    (image1,image2) = (getFilteredImage(image,sigmaX=3,sigmaY=1.5),getFilteredImage(image,sigmaX=4.5,sigmaY=1.5) )
    vector = getFeatureVector(image1,image2)
    return vector

# This is similar to processImage(), but since we want to define several templates which
# denotes the rotation angles for each training image, each training image has to
# be converted into several vector.
def processImageWithRotation(fileName,degree):
    img = cv2.imread(fileName)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (innerCircle,outterCircle) = irisLocalization(image)
    image = getNormalization(image,innerCircle,outterCircle)
    image = getRotation(image,degree)
    image = allEnhancement(image)
    (image1,image2) = (getFilteredImage(image,sigmaX=3,sigmaY=1.5),getFilteredImage(image,sigmaX=4.5,sigmaY=1.5) )
    vector = getFeatureVector(image1,image2)
    return vector



# This function loops through all the files in our database and transfer every
# image into a vector
def getDatabase(folder):
    number = folder + 2
    folder = str(folder)
    vec = []
    # Folder 1 contains training image, so we do this with rotation
    if folder =='1':
        rotation = [-9,-6,-3,0,3,6,9]
        for i in range(1,109):
            for j in range(1,number+1):
                thisFileName = './CASIA Iris Image Database (version 1.0)/'
                index = "%03d" % (i,)
                fileName = thisFileName + index + '/'+folder+'/' + index +'_'+folder+'_%d' %(j) +'.bmp'
                print(fileName)
                for q in range(7):
                    vec.append(processImageWithRotation(fileName,rotation[q]/3))
    # Folder 2 doesn't have to do rotation
    else:
        for i in range(1,109):
            for j in range(1,number+1):
                thisFileName = './CASIA Iris Image Database (version 1.0)/'
                index = "%03d" % (i,)
                fileName = thisFileName + index + '/'+folder+'/' + index +'_'+folder+'_%d' %(j) +'.bmp'
                print(fileName)
                vec.append(processImage(fileName))
    return vec
        

# We do LDA for each vector and use l1,l2,and cosine distance to calculate the
# similarity between pictures. This function takes training and testing data and
# output the accuracy rate for our matching
def getMatching(train,test,LDADimention=107,distanceMeasure=3):
    trainX = np.array(train)
    testX  = np.array(test)
    irisY = np.arange(1,109)
    trainY = np.repeat(irisY,3*7)
    testY = np.repeat(irisY,4)
    trainClass = np.repeat(irisY,3)
    
    clf = LDA(n_components = LDADimention)
    clf.fit(trainX,trainY)
    newTrain = clf.transform(trainX)
    newTest = clf.transform(testX)
    
    
    predicted = np.zeros(testX.shape[0])
    for i in range(testX.shape[0]):
        vec = np.zeros(int(trainX.shape[0]/7))
        thisTest = newTest[i:i+1]
        for j in range(len(vec)):
            distance = np.zeros(7)
            for q in range(7):
                if distanceMeasure ==3:
                    distance[q] = scipy.spatial.distance.cosine(thisTest,newTrain[j*7+q:j*7+q+1])
                elif distanceMeasure ==1:
                    distance[q] = scipy.spatial.distance.cityblock(thisTest,newTrain[j*7+q:j*7+q+1])
                else:
                    distance[q] = scipy.spatial.distance.sqeuclidean(thisTest,newTrain[j*7+q:j*7+q+1])
                
            vec[j] = np.min(distance)
        shortestDistanceIndex = np.argmin(vec)
        predicted[i] = trainClass[shortestDistanceIndex]
    
    predicted = np.array(predicted,dtype =np.int)
    accuracyRate = 1 - sum(predicted != testY)/len(testY)
    return accuracyRate


