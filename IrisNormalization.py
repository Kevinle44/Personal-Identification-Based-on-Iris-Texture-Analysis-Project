
# Iris Normalization
# After getting the boundary of pupil and iris, normalize the iris part which is
# a round image into a rectangle image
# Author: Zonghao Li   (zl2613@columbia.edu)
# Written in: 2018/04/04




import numpy as np

# return the distance between two points
def getDistance(x1,y1,x2,y2):
    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )

# return the inverse of tangent between two points
def getInverseTan(x1,y1,x2,y2):
    tanTheta = (y2 - y1) / (x2 - x1)
    return np.arctan(tanTheta)

# return the radius of the iris, which is longer than the radius of pupil
def getLongRadius(d1, r2, theta):
    x1 = (2*d1*np.cos(theta) + np.sqrt( (2*d1*np.cos(theta))**2 - 4*(d1*d1 - r2*r2) )) / 2
    return x1

# transfer the round image into a rectangle image, this function takes the X,Y of the
# rectangle image and finds the corresponding x,y from the round image
def getxy(X,Y,innerCircle,outterCircle):
    
    
    (M, N) = (64, 512)
    theta = 2 * np.pi * X / N 
    (x1, y1, r1) = (innerCircle[0], innerCircle[1], innerCircle[2])
    (x2, y2, r2) = (outterCircle[0], outterCircle[1], outterCircle[2])
    d1 = getDistance(x1,y1,x2,y2)
    diffTheta = getInverseTan(x1, y1, x2, y2)
    
    longRadius = getLongRadius(d1, r2, diffTheta)
    xInner = x1 + r1 * np.cos(theta)
    yInner = y1 + r1 * np.sin(theta)
    
    xOutter = x1 + longRadius * np.cos(theta)
    yOutter = y1 + longRadius * np.sin(theta)
    
    x = int(xInner + (xOutter - xInner) * Y / M)
    y = int(yInner + (yOutter - yInner) * Y / M)
    
    x = min(319,x) or max(0,x)
    y = min(279,y) or max(0,y)
    
    return(x, y)
    


# For each pixel in normalized image, find the value for the corresponding
# pixels in the original image and fill in the value
    
def getNormalization(image,innerCircle,outterCircle):
    newImage = np.zeros((64,512))

    for Y in np.arange(64):
        for X in np.arange(512):
            (x, y) = getxy(X, Y,innerCircle,outterCircle)
            newImage[Y, 511- X] = image[y, x]
    return newImage
 

# This function takes normalized image and rotate the rectangle image to specified
# degree

def getRotation(image,degree):
    pixels = abs(int(512*degree/360))
    if degree > 0:
        return np.hstack([image[:,pixels:],image[:,:pixels]] )
    else:
        return np.hstack([image[:,(512 - pixels):],image[:,:(512 - pixels)]] )
 
 