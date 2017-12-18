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

# initialize background substractor object
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# initialize the SURF object & set Hessian Threshold to 400
surf = cv2.xfeatures2d.SURF_create(400)

# boolean used to indicate if we have already detected a pedestrian or not. If yes, we skip the function call
# to hogSVMDetection
PED_ALREADY_DET = False

# int used to know if the current frame is the reference frame for the BruteForce Matching or not. If yes, this means
# we cannot do a matching yet, and we just have to store the rectangle, keypoints & descriptors
DETECTION_COUNT = 0

# initial margin values
xMarginStart = 35
yMarginStart = 50

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
#trackingImages = trackingImages[60:]
# total number of frames
NbTotImages = len(trackingImages)

# loop over the image paths
for imagePath in trackingImages:

    imagePath = data_path + imagePath
    print(imagePath)

    # load the image and resize it to reduce detection time and improve detection accuracy
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(480, image.shape[1]))
    disp_image = image.copy()
    image = backgroundSubstraction(fgbg, image)

    if not PED_ALREADY_DET:  # no pedestrian has been detected
        rects = hogSVMDetection(hog, image)

    if len(rects) > 0:  # at least one pedestrian has been detected
        print('Pedestrian detected')
        (xA, yA, xB, yB) = rects[0]  # Get the region coord. to search for a pedestrian (corresponds to previous frame).
        # only care about the first pedestrian for now

        print('(xA, yA, xB, yB) = ', (xA, yA, xB, yB))
        pedestrian = image[yA: yB, xA: xB]  # select the region to search for the keypoints

        print('Number of previous keypoints : ', len(previousKeypoints))
        print('Number of associated descriptors : ', previousDescriptors.shape)

        # find keypoints and descriptors using SURF
        currentKeypoints, currentDescriptors = surf.detectAndCompute(pedestrian, None)
        print('Number of current keypoints found: ', len(currentKeypoints))
        print('Number of associated descriptors : ', currentDescriptors.shape)

        currentKeypoints = updateKeypointsCoordinates(currentKeypoints, xA, yA)
        print('Keypoints coordinates updated')

        allCurrentKeypoints = currentKeypoints
        allCurrentDescriptors = currentDescriptors

        if DETECTION_COUNT == 0:
            print('This is the reference frame. Not performing Brute Force Matching & LeastSquare.')
            # update the keypoints coordinates
            #currentKeypoints = updateKeypointsCoordinates(currentKeypoints, xA, yA)
            #print('Keypoints coordinates updated')

        else:
            print('We should have some reference from the previous frame. Performing BFMatching & LeastSquare.')
            # perform brute force matching
            previousKeypoints, previousDescriptors, currentKeypoints, currentDescriptors = bruteForceMatching(
                previousKeypoints, previousDescriptors, currentKeypoints, currentDescriptors)
            print('Number of matched current Keypoints : ', len(currentKeypoints))

            # update the keypoints coordinates
            #currentKeypoints = updateKeypointsCoordinates(currentKeypoints, xA, yA)
            #print('Keypoints coordinates updated')

            # we now should perform the least square regression
            #sx, sy, tx, ty = leastSquareRegression(previousKeypoints, currentKeypoints)
            #theta, alpha, tx, ty = leastSquareRegression2D(previousKeypoints, currentKeypoints)
            #M = homographyMatrix(previousKeypoints, currentKeypoints)
            tx, ty = findTranslationTransf(previousKeypoints, currentKeypoints)
            theta = 0
            alpha = 1
            #print('tx, ty =', tx, ty)
            #print('sx, sy, tx, ty = ', sx, sy, tx, ty)
            print('theta, alpha, tx, ty = ', theta, alpha, tx, ty)
            #print('homography matrix M = ', M)
            #print(previousKeypoints[0].pt, currentKeypoints[0].pt)


            # update the bounding rectangle coordinates
            #xA, yA, xB, yB = updateRectangleLeastSquare(rectCoord=(xA,yA,xB,yB), scaling=(sx,sy), translation=(tx,ty))
            xA, yA, xB, yB = updateRectangleLeastSquare2D((xA,yA,xB,yB), theta, alpha, tx, ty)
            #xA, yA, xB, yB = updateRectangleHomography((xA, yA, xB, yB), M)

            # cast to int
            xA = int(xA)
            xB = int(xB)
            yA = int(yA)
            yB = int(yB)

        # save the new bounding rectangle coordinates
        rects[0] = (xA, yA, xB, yB)

        # save all keypoints detected in this frame as the previous ones
        previousKeypoints = allCurrentKeypoints

        # save associated descriptors
        previousDescriptors = allCurrentDescriptors
        print('previousDescriptors : ', previousDescriptors.shape)

        # draw the bounding rectangle & keypoints
        cv2.rectangle(disp_image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        disp_image = cv2.drawKeypoints(disp_image, currentKeypoints, disp_image)

        PED_ALREADY_DET = True
        DETECTION_COUNT += 1
        print('DETECTION_COUNT = ', DETECTION_COUNT)

    cv2.imshow("Tracking", disp_image)
    cv2.waitKey(300)  # display images at roughly 15 fps
    cv2.destroyAllWindows()
    print('\n')
