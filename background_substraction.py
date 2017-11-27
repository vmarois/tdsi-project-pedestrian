import cv2
import numpy as np
import os


# function to subtract the background using Gaussian Mixture-based Background/Foreground Segmentation Algorithm
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# elliptic kernel for morpho math
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


data_path = 'data/'
# first, get the list of the images located in the data_path folder. These images names (e.g. 'tracking_0001.jpeg') will
# be used for indexing.
trackingImages = [name for name in os.listdir(os.path.join(os.curdir, "data/")) if not name.startswith('.')]
# We sort this list to get the names in increasing order
trackingImages.sort(key=lambda s: s[10:13])
trackingImages.sort()


# loop over the image paths
for imagePath in trackingImages:
    imagePath = data_path + imagePath
    # load the image and resize it to reduce detection time and improve detection accuracy
    image = cv2.imread(imagePath)

    # apply the  background subtraction algorithm
    fgmask = fgbg.apply(image)

    # filter the bushes on the mask (maybe not necessary)
    fgmask = (cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2))/255
    fgmask = (cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=5))


    # apply the mask on the image
    image[:, :, 0] = image[:, :, 0] * fgmask
    image[:, :, 1] = image[:, :, 1] * fgmask
    image[:, :, 2] = image[:, :, 2] * fgmask

    # display result
    cv2.imshow("Tracking", image)
    cv2.waitKey(0)
