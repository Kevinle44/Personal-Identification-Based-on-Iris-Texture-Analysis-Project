readme file for GR5293 Project Iris Recognition
Author: Zonghao Li   (zl2613@columbia.edu)
Written in: 2018/04/04
For further updates on this project, please check: https://github.com/zonghao1/Personal-Identification-Based-on-Iris-Texture-Analysis-Project

This project implemented the Iris Recognition Algorithm Personal Identification Based on Iris Texture Analysis by Li Ma et al.
Paper avaiable at    http://ieeexplore.ieee.org/document/1251145/

#Portions of the research in this paper use the CASIA-IrisV1 collected by the Chinese Academy of Sciences' Institute of Automation (CASIA)” and a reference to “CASIA-IrisV1, http://biometrics.idealtest.org/
——————————————————————————————————————————
Code Reproduce

Please put all python files and the CASIA Iris Image Database (version 1.0)(avaiable at http://biometrics.idealtest.org/) into python working directory and make sure libraries like bumpy, scipy, sklearn, and cv2 was installed previously. 

IrisRecognition is the main file. Run main() would run all the algorithm step by step including IrisLocalization,IrisNormalization, ImageEnhancement, Feature Extraction, IrisMatching, and PerformanceEnvaluation to give the result plots.

Since getting the dataset takes a long time, I wrote another function called runAllReduced() which directly uses the vector data I saved before to do IrisMatching and PerformanceEnvaluation. So change runAll() in main function into runAllReduced() would save running time. Saved data “irisTest.npy” and “irisTrain.npy” was included in “additional file” folder.

“IrisManualCheck.ipynb” was also included in the “additional file” folder. This could plot each iris image after localization and we could check whether the boundary we found is good or not. 

The outputs ( “Recognition results using features of different dimensionality” (Figure 10 in paper), “Recognition results using different similarity measures” (Table3 in paper), and “Recognition results using different dimensionality of PCA” was included in the output
folder).

——————————————————————————————————————————
General Procedures
I. IrisLocalization.py 

a).First we try to localize the pupil. Since the pupil is darker than other parts in the image, we estimate the center of pupil by finding the row and column that has minimum sum of values. In some images the eyelash was too dark that influenced the outcome, so I fist took a sub image so that the influence of eyelash was greatly minimized. 
b). After the first rough estimate of the center of pupil, we binarize a 120*120 region centered at that with a threshold of 65 which is the optimal value after several testing. From this binary image, we could have a better estimate of the pupil center and also pupil 
radius.
c). Then we use Hough Transformation to find the boundary of pupil and Iris. For pupil, we pass in the estimate of radius we calculated in part b. For Iris, we pass in minRadius=98 and maxRadius=118 which is the optimal parameters after long time testing.

II. IrisNormalization.py
a).After getting the boundary of pupil and iris, normalize the iris part which is a round image into a rectangle image. For each pixel in normalized image, find the value for the corresponding pixels in the original image and fill in the value. 
b). I also wrote the rotation function in this part. Since for a specific sample, the iris image we took may rotate slightly, so we use the normalized image and rotate it according to the degrees we specified as the training set.

III. ImageEnhancement.py
a). The paper proposed to estimates the background value for each 16*16 small block and subtract that from the normalized image, but after testing this method is not useful for our accuracy rate, so I didn't use this in next steps. 
b). For each 32*32 pixels region, I did image enhancement by histogram equalization and use this image for further processing.

IV. FeatureExtraction.py
a). Since upper portion of the normalized iris image provides most useful texture information proposed in the paper, we take the top 48 rows as Region of Interest. 
b). They use even-symmetric Gabor filter to do convolution with the image. The paper used two set of parameters SigmaX1 = 3,SigmaY1 = 1.5, SigmaX2 = 4.5, SigmaY2 = 1.5. With these two sets of parameters, I calculated two Gabor filter and thus get two set of filtered image.
c). After that we took the two filtered Image and extracts mean and standard deviation
for each 8*8 small block as the feature vector.

V. IrisMatching.py
a). After all the previous parts, an image would be transferred into a feature vector. This file uses all functions before and transfers the image database we have into vector database. For each training image, we get 7 vectors since we do rotation for 7 degrees, and for each test image we get 1 vector.
b). Then we apply Linear Discriminant Analysis (LDA) on our dataset and then calculate the L1,L2, and cosine distance and use the class of minimum	distance as the predicted distance. After several test, cosine distance performs better than the other two, so I used cosine distance in further parts. 

VI. PerformanceEnvaluation.py
a). For the Recognition results using features of different dimensionality (Figure10 in paper), I wrote a function called getCRRCurve() which could output this plot.
b). For the Recognition results using different similarity measures (Table3 in paper), I wrote a function called getTable().
c). I also applied PCA for dimension reduction. Within each PCA dimension, the maximum accuracy rate was calculated by trying LDA dimensions of 90,100,107 which approves to be the dimensions with highest accuracy rate in general. The result for single image accuracy was silightly increase from accuracy rate 88.41% to 90.50% after applying PCA. The result for each sample accuracy is 97.22%.

VII. IrisRecognition.py
a).This is the main file for the whole project. Simply run the main function would run all the algorithm step by step including IrisLocalization,IrisNormalization, ImageEnhancement, Feature Extraction, IrisMatching, and PerformanceEnvaluation to give the result plots.

b). Since getting the dataset takes a long time, I wrote another function called runAllReduced() which directly uses the vector data I saved before to do IrisMatching and PerformanceEnvaluation. So change runAll() in main function into runAllReduced() would save running time. 


—————————————————————————————————————————
Limitation

Currently the best accuracy rate this algorithm could achieve is 90.50%. By further tuning the parameters in IrisLocalization.py, we can get a better accuracy rate. Other PCA dimension and LDA dimensions could also be tested to see if we could achieve a better accuracy rate. Also, using bootstrap to generate dataset could give us an estimate and Confidence Interval of the variance of the CRR, FMR, and FNMR. 

Another limitation of this project is the sample size. We only have 108 samples with 7 images for each sample. If we want to further test this algorithm, more samples would give us a much better results. 






