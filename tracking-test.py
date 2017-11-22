# this file is part of tdsi-pedestrian

import cv2
import numpy as np
import imutils
import os

from pedtracking import *

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
# set the Support Vector Machine to be pre-trained pedestrian detector
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# initialize the SURF object & set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(400)

# initialize the BruteForceMatcher object with default params
bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)

# boolean used to indicate if we have already detected a pedestrian or not. If yes, we skip the function call
# to hogSVMDetection
PED_ALREADY_DET = False

# int used to know if the current frame is the reference frame for the BruteForce Matching or not. If yes, this means
# we cannot do a matching yet, and we just have to store the rectangle, descriptors
DETECTION_COUNT = 0

# empty list to store the previous keypoints
previousKeypoints = []

# empty numpy.ndarray to store the previous descriptors
previousDescriptors = np.empty((0, 0))

data_path = 'data/'
# first, get the list of the images located in the data_path folder. These images names (e.g. 'tracking_0001.jpeg') will
# be used for indexing.
trackingImages = [name for name in os.listdir(os.path.join(os.curdir, "data/")) if not name.startswith('.')]
# We sort this list to get the names in the correct order
trackingImages.sort(key=lambda s: s[10:13])


# loop over the image paths
for imagePath in trackingImages:

    imagePath = data_path + imagePath
    print(imagePath)

    # load the image and resize it to reduce detection time and improve detection accuracy
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(480, image.shape[1]))

    if not PED_ALREADY_DET:  # no pedestrian has been detected
        rects = hogSVMDetection(hog, image)

    if len(rects) > 0:  # at least one pedestrian has been detected
        print('Pedestrian detected')
        (xA, yA, xB, yB) = rects[0]  # Get the region coordinates to search for a pedestrian
        # (corresponds to previous frame).
        # only care about the first pedestrian for now

        print('(xA, yA, xB, yB) = ', (xA, yA, xB, yB))
        pedestrian = image[yA: yB, xA: xB]  # select the region to search for the keypoints

        print('Number of previous keypoints : ', len(previousKeypoints))
        print('Number of associated descriptors : ', previousDescriptors.shape)

        # find keypoints and descriptors using SURF
        currentKeypoints, currentDescriptors = surf.detectAndCompute(pedestrian, None)
        print('Number of current keypoints found: ', len(currentKeypoints))
        print('Number of associated descriptors : ', currentDescriptors.shape)

        if DETECTION_COUNT == 0:
            print('This is the reference frame. Not performing Brute Force Matching.')
            # update the keypoints coordinates
            for keypoint in currentKeypoints:
                (x, y) = keypoint.pt
                x += xA
                y += yA
                keypoint.pt = (x, y)
            print('Keypoints coordinates updated')

        else:
            print('We should have some reference from the previous frame. Performing BFMatching.')
            currentKeypoints, currentDescriptors = bruteForceMatching(previousKeypoints,
                                                                      previousDescriptors,
                                                                      currentKeypoints,
                                                                      currentDescriptors)
            # update the keypoints coordinates
            for keypoint in currentKeypoints:
                (x, y) = keypoint.pt
                x += xA
                y += yA
                keypoint.pt = (x, y)
            print('Keypoints coordinates updated.')

        # save the current keypoints as the previous ones
        previousKeypoints = currentKeypoints

        # save the current descriptors as the previous ones
        previousDescriptors = currentDescriptors
        print('previousDescriptors : ', type(previousDescriptors), ' ', previousDescriptors.shape)
        # update the bounding rectangles coordinates and save them
        xA, yA, xB, yB = updateRectangle(currentKeypoints, delta=30)
        rects[0] = (xA, yA, xB, yB)

        # draw the bounding rectangle & keypoints
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        image = cv2.drawKeypoints(image, currentKeypoints, image)

        PED_ALREADY_DET = True
        DETECTION_COUNT += 1

    cv2.imshow("{}".format(imagePath), image)
    cv2.waitKey(1)  # display images at roughly 15 fps
    cv2.destroyAllWindows()
    print('\n')