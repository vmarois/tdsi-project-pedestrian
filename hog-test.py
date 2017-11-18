# this file is part of tdsi-pedestrian

import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression
import os

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
# set the Support Vector Machine to be pre-trained pedestrian detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# initialize the SURF object & set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(400)

data_path = 'data/'
# first, get the list of the images located in the data_path folder. These images names (e.g. 'tracking_0001.jpeg') will
# be used for indexing.
trackingImages = [name for name in os.listdir(os.path.join(os.curdir, "data/")) if not name.startswith('.')]
# We sort this list to get the names in increasing order
trackingImages.sort(key=lambda s: s[10:13])


# loop over the image paths
for imagePath in trackingImages:

    imagePath = data_path + imagePath
    # load the image and resize it to reduce detection time and improve detection accuracy
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(480, image.shape[1]))

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.03)
    # the 'scale'  is one of the most important parameters to tune when performing pedestrian detection

    # if one (or more) pedestrian has been detected
    if len(rects) > 0:
        # apply non-maxima suppression to the bounding boxes using a fairly large overlap threshold to try to maintain
        # overlapping boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # loop over the detected pedestrian
        for (xA, yA, xB, yB) in rects:
            # select the roi corresponding to the pedestrian detected
            pedestrian = image[yA: yB, xA: xB]

            # find keypoints and descriptors using SURF
            kp, des = surf.detectAndCompute(pedestrian, None)
            # seem like we have about 20 keypoints per pedestrian

            # draw keypoints on the pedestrian
            img = cv2.drawKeypoints(pedestrian, kp, pedestrian)

            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show some information on the number of bounding boxes
        print("[INFO] {}: {} after suppression".format(imagePath, len(rects)))

    # show the output images
    cv2.imshow("{}".format(imagePath), image)
    cv2.waitKey(1)  # display images at roughly 15 fps
    cv2.destroyAllWindows()

