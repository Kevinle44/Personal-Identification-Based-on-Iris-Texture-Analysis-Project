import numpy as np
import scipy.signal

def roi(image):
    return image[0:48,:]
    

def m1(x,y,sigmaY):
    f = 1/sigmaY
    return np.cos(2 *np.pi *f *np.sqrt(x**2 + y**2) )
    
def g(x,y,sigmaX,sigmaY):
    return (1/(2*np.pi*sigmaX*sigmaY) * np.exp( -(x**2/sigmaX**2 + y**2/sigmaY**2)/2) * m1(x,y,sigmaY) )
    
def getKernal(sigmaX,sigmaY):
    kernal = np.zeros((9,9))
    for row in range(9):
        for col in range(9):
            kernal[row,col] = g( (-4+col),(-4+row),sigmaX,sigmaY)
    return kernal

sigmaX1 = 3
sigmaY1 = 1.5
sigmaX2 = 4.5
sigmaY2 = 1.5


def FakegetFilteredImage(image,sigmaX,sigmaY):
    newImage = image.copy()
    thisKernal = getKernal(sigmaX,sigmaY)
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            
            newImage[row,col] = signal.convolve2d(image,kernal)[row,col]
    return newImage

def getFilteredImage(image,sigmaX,sigmaY):
    image = roi(image)
    thisKernal = getKernal(sigmaX,sigmaY)
    newImage = scipy.signal.convolve2d(image,thisKernal,mode='same')
    return newImage



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
