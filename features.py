import cv2
import numpy as np
from matplotlib import pyplot as plt



path_left = '/Volumes/LABDATA/SLAM/2011_09_26/2011_09_26_drive_0061_sync/image_00/data/'
path_right = '/Volumes/LABDATA/SLAM/2011_09_26/2011_09_26_drive_0061_sync/image_01/data/'


image_idx = 0
filename_left = path_left + "%.10i"%image_idx + ".png"
filename_right = path_right + "%.10i"%image_idx + ".png"


img1 = cv2.imread(filename_left,0)    
# img2 = cv2.imread(filename_right,0) 

# # Initiate SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)

# plt.imshow(img3),plt.show()

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector()

# find and draw the keypoints
kp = fast.detect(img1,None)
img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

# Print all default params
print ("Threshold: ", fast.getInt('threshold'))
print ("nonmaxSuppression: ", fast.getBool('nonmaxSuppression'))
print ("neighborhood: ", fast.getInt('type'))
print ("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
fast.setBool('nonmaxSuppression',0)
kp = fast.detect(img,None)

print ("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

cv2.imwrite('fast_false.png',img3)