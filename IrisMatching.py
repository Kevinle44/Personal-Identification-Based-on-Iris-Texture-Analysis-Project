import numpy as np
import cv2
from sklearn.neighbors.nearest_centroid import NearestCentroid

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from IrisLocalization  import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *

def processImage(fileName):
    img = cv2.imread(fileName)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (innerCircle,outterCircle) = irisLocalization(image)
    image = getNormalization(image,innerCircle,outterCircle)
    image = allEnhancement(image)
    (image1,image2) = (getFilteredImage(image,sigmaX=3,sigmaY=1.5),getFilteredImage(image,sigmaX=4.5,sigmaY=1.5) )
    vector = getFeatureVector(image1,image2)
    return vector


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




def getDatabase(folder):
    number = folder + 2
    folder = str(folder)
    vec = []
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
    else:
        for i in range(1,109):
            for j in range(1,number+1):
                thisFileName = './CASIA Iris Image Database (version 1.0)/'
                index = "%03d" % (i,)
                fileName = thisFileName + index + '/'+folder+'/' + index +'_'+folder+'_%d' %(j) +'.bmp'
                print(fileName)
                vec.append(processImage(fileName))
    return vec
        

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
    newPre = []
    for i in range(108):
        newPre.append(scipy.stats.mode(predicted[i*4:i*4+4])[0][0])
    accuracyRate = 1 - sum(newPre != irisY)/len(irisY)
    return accuracyRate





