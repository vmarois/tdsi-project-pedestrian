import cv2
import numpy as np

img = cv2.imread('data/tracking_0250.jpeg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SURF_create(500)
kp = sift.detect(gray,None)
print('{}'.format(kp))
img=cv2.drawKeypoints(img,kp,img)

cv2.imwrite('surf_keypoints2.jpg',img)
cv2.imshow('sift_example',img)