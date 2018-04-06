
# Feature Extraction
# Extract features using even-symmetric Gabor filters from enhanced image from last
# part
# Author: Zonghao Li   (zl2613@columbia.edu)
# Written in: 2018/04/04


import numpy as np
import scipy.signal



# get the Region Of Interest which is the top 48 rows
def roi(image):
    return image[0:48,:]
    
# m1 and g are helper functions for parameters in Gabor filters
def m1(x,y,sigmaY):
    f = 1/sigmaY
    return np.cos(2 *np.pi *f *np.sqrt(x**2 + y**2) )
    
def g(x,y,sigmaX,sigmaY):
    return (1/(2*np.pi*sigmaX*sigmaY) * np.exp( -(x**2/sigmaX**2 + y**2/sigmaY**2)/2) * m1(x,y,sigmaY) )
    
    
# This function calculates the Gabor Filter with specified siamgeX,sigmaY    
def getKernal(sigmaX,sigmaY):
    kernal = np.zeros((9,9))
    for row in range(9):
        for col in range(9):
            kernal[row,col] = g( (-4+col),(-4+row),sigmaX,sigmaY)
    return kernal

# This function calculates the convolve of image and filter
def getFilteredImage(image,sigmaX,sigmaY):
    image = roi(image)
    thisKernal = getKernal(sigmaX,sigmaY)
    newImage = scipy.signal.convolve2d(image,thisKernal,mode='same')
    return newImage


# This function takes the two filtered Image and extracts mean and standard deviation
# for each 8*8 small block as the feature vector for a specific image
def getFeatureVector(f1,f2):
    nrow = int(f1.shape[0]/8)
    ncol = int(f1.shape[1]/8)
    vec = np.zeros(nrow*ncol*2*2)
    for i in range(2):
        image = [f1,f2][i]
        for row in range(nrow):
            for col in range(ncol):
                meanValue = np.mean( np.abs( image[row*8: (row+1) * 8,col*8: (col +1) * 8] ))
                sdValue = np.sum(abs(image[row*8: (row+1) * 8,col*8: (col +1) * 8] - meanValue))/ (8*8)
                vec[i*768 + 2*row*ncol + 2*col] = meanValue
                vec[i*768 + 2*row*ncol + 2*col + 1] = sdValue
    return vec   
