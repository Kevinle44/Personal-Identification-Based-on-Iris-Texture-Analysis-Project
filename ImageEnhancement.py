import numpy as np
import cv2

def backgroundEstimate(image):
    nrow = int(image.shape[0]/16)
    ncol = int(image.shape[1]/16)
    newImage = np.repeat(np.NaN,repeats = nrow*ncol)
    newImage = newImage.reshape(nrow,ncol)
    for row in range(nrow):
        for col in range(ncol):
            allrows = np.arange(row*16, (row+1) * 16)
            allcols = np.arange(col*16, (col +1) * 16)
            value = np.mean(image[row*16: (row+1) * 16,col*16: (col +1) * 16])
            newImage[row,col] = value
    res = cv2.resize(newImage,None,fx=16, fy=16, interpolation = cv2.INTER_CUBIC)
    return res

def subtractBackground(image):
    return image-backgroundEstimate(image) 



def enhancement(image):
    image = np.array(image,dtype=np.uint8)
    image = cv2.equalizeHist(image)
    return image


def imageEnhancementByParts(image):
    nrow = int(image.shape[0]/32)
    ncol = int(image.shape[1]/32)
    for row in range(nrow):
        for col in range(ncol):
            enhanced = enhancement(image[row*32:(row+1)*32,col*32:(col+1)*32])
            image[row*32:(row+1)*32,col*32:(col+1)*32] = enhanced
    return image


def allEnhancement(image):
#    image = subtractBackground(image)
    image = imageEnhancementByParts(image)
    return image

