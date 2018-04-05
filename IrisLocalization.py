#################################################
# Iris Localization
# Read the image data and output the boundary for the pupil and iris
# Author: Zonghao Li   (zl2613@columbia.edu)
# Written in: 2018/04/04

#################################################

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image data and output the boundary for the pupil and iris
def irisLocalization(image):
   
    #Take a subImage and find the location that has lowest row and column sum
    subImage = image[60:240,100:220]
    vertical = subImage.sum(axis = 0)
    horizontal = subImage.sum(axis = 1)
    
    yp = np.argmin(horizontal) +60 
    xp = np.argmin(vertical) + 100
    
    #Use a threshhold of 64 to localize the pupil
    region120 = image[yp-60:yp+60,xp-60:xp+60]
    ret,th1 = cv2.threshold(region120,64,65,cv2.THRESH_BINARY)

    #Based on the binary image, re-calculate the center of the pupil and estimate
    # the radius of the pupil
    mask1 = np.where(th1>0,1,0)

    vertical = mask1.sum(axis = 0)
    horizontal = mask1.sum(axis = 1)

    minyp = np.argmin(horizontal) 
    minxp = np.argmin(vertical)

    radius1 = (120 - sum(mask1[minyp])) / 2
    radius2 = (120 - np.sum(mask1,axis=0)[minxp]) / 2
    radius = int((radius1 + radius2) /2)

    
    newyp = np.argmin(horizontal) + yp - 60
    newxp = np.argmin(vertical) + xp - 60

    #print(newyp,newxp)
    
    # Now look for the out boundary
    
    # First get a smaller image to accelarate
    region240 = image[np.arange(newyp-120, min(279, newyp+110)),:][:,np.arange(newxp-135,min(319,newxp+135))]
    region120 = image[np.arange(newyp-60, min(279, newyp+60)),:][:,np.arange(newxp-60,min(319,newxp+60))]
    
    
    # Using Hough transform for detecting the circle for pupil

    for loop in range(1,5):
        circles = cv2.HoughCircles(region120,cv2.HOUGH_GRADIENT,1,250,
                           param1=50,param2=10,minRadius=(radius-loop),maxRadius=(radius+loop))
        if type(circles) != type(None):
            break
        else:
            pass
    circles = np.around(circles)


    # Using Hough transform for detecting the circle for iris
    circles1 = cv2.HoughCircles(region240, cv2.HOUGH_GRADIENT,1,250,
                           param1=30,param2=10,minRadius=98,maxRadius=118)
    circles1 = np.around(circles1)                           

    
    # return the output and draw the boundary
    image1 = image.copy()

    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(image1,( int(i[0]+ newxp - 60),int(i[1] + newyp - 60)),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(image1,( int(i[0]+ newxp - 60),int(i[1] + newyp - 60)),i[2],(0,255,0),2)
        innerCircle = [i[0] + newxp - 60,i[1] + newyp -60 ,i[2]]


    for i in circles1[0,:]:
    # draw the outer circle
        cv2.circle(image1,( int(i[0]+ newxp - 135),int(i[1] + newyp - 120)),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(image1,( int(i[0]+ newxp - 135),int(i[1] + newyp - 120)),i[2],(0,255,0),3)
        outterCircle = [int(i[0]+ newxp - 135),int(i[1] + newyp - 120),i[2]   ]
        
    return(innerCircle,outterCircle)




# This function is almost the same as irisLocalization(), but this also draws the
# image with two boundaries. Designed for testing, debugging, and visualizing.

def irisLocalizationDrawing(fileName):
    #Take a subImage and find the location that has lowest row and column sum
    img = cv2.imread(fileName)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    subImage = image[60:240,100:220]
    vertical = subImage.sum(axis = 0)
    horizontal = subImage.sum(axis = 1)
    
    yp = np.argmin(horizontal) +60 
    xp = np.argmin(vertical) + 100
    
    #Use a threshhold of 50 to localize the pupil
    region120 = image[yp-60:yp+60,xp-60:xp+60]
    ret,th1 = cv2.threshold(region120,64,65,cv2.THRESH_BINARY)


    mask1 = np.where(th1>0,1,0)

    vertical = mask1.sum(axis = 0)
    horizontal = mask1.sum(axis = 1)

    minyp = np.argmin(horizontal) 
    minxp = np.argmin(vertical)

    radius1 = (120 - sum(mask1[minyp])) / 2
    radius2 = (120 - np.sum(mask1,axis=0)[minxp]) / 2
    radius = int((radius1 + radius2) /2)
#    radius = max(radius,39)
    
    newyp = np.argmin(horizontal) + yp - 60
    newxp = np.argmin(vertical) + xp - 60

#    print(newyp,newxp)
    # Now look for the out boundary
    # First get a smaller image to accelarate
    region240 = image[np.arange(newyp-120, min(279, newyp+110)),:][:,np.arange(newxp-135,min(319,newxp+135))]
    region120 = image[np.arange(newyp-60, min(279, newyp+60)),:][:,np.arange(newxp-60,min(319,newxp+60))]


    for loop in range(1,5):
        circles = cv2.HoughCircles(region120,cv2.HOUGH_GRADIENT,1,250,
                           param1=50,param2=10,minRadius=(radius-loop),maxRadius=(radius+loop))
        if type(circles) != type(None):
            break
        else:
            pass
    circles = np.around(circles)


    circles1 = cv2.HoughCircles(region240, cv2.HOUGH_GRADIENT,1,250,
                           param1=30,param2=10,minRadius=98,maxRadius=118)
    circles1 = np.around(circles1)                           


    image1 = image.copy()

    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(image1,( int(i[0]+ newxp - 60),int(i[1] + newyp - 60)),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(image1,( int(i[0]+ newxp - 60),int(i[1] + newyp - 60)),i[2],(0,255,0),2)
        innerCircle = [i[0] + newxp - 60,i[1] + newyp -60 ,i[2]]


    for i in circles1[0,:]:
    # draw the outer circle
        cv2.circle(image1,( int(i[0]+ newxp - 135),int(i[1] + newyp - 120)),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(image1,( int(i[0]+ newxp - 135),int(i[1] + newyp - 120)),i[2],(0,255,0),3)
        outterCircle = [int(i[0]+ newxp - 135),int(i[1] + newyp - 120),i[2]   ]
    
    plt.imshow(image1,cmap='gray')
    plt.show()

    
    return(innerCircle,outterCircle)