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
RESTART_SVM_DET = 20

# int used to know if the current frame is the reference frame for the BruteForce Matching or not. If yes, this means
# we cannot do a matching yet, and we just have to store the rectangle, keypoints & descriptors
DETECTION_COUNT = 0

# int used to perform HOG + SVM detection every 10 frames
SVMDETECT = 0

# empty list to store rectangle coordinates
rects = []

# empty list to store keypoints from all pedestrians
allKeypoints = []

# empty list to store the previous keypoints
previousKeypoints = []

# empty list to store descriptors from all pedestrians
allDescriptors = []

# empty numpy.ndarray to store the previous descriptors
previousDescriptors = np.empty((0, 0))

data_path = 'data/'
# first, get the list of the images located in the data_path folder. These images names (e.g. 'tracking_0001.jpeg') will
# be used for indexing.
trackingImages = [name for name in os.listdir(os.path.join(os.curdir, "data/")) if not name.startswith('.')]
# We sort this list to get the names in the correct order
trackingImages.sort(key=lambda s: s[10:13])
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

        if not SVMDETECT == 10:
            SVMDETECT += 1
        else:
            print('Performing HOG + SVM pedestrian detection')
            rects = hogSVMDetection(hog, image)
            SVMDETECT = 0

    #### UNCOMMENT THE NEXT LINE TO PERFORM MORE THAN ONE TRACKING
    '''
    else:
        if RESTART_SVM_DET != 0:
            RESTART_SVM_DET -= 1
        else:
            print('Performing HOG + SVM detection to check for new pedestrians')
            newrects = hogSVMDetection(hog, image)
            RESTART_SVM_DET = 20
            rects = np.append(rects[:], newrects, axis=0)
    '''
    # Remove coordinates of rect already detected
    rects = removeRedundantRectangles(rects)
    print(len(rects))

    if len(rects) > 0: # at least one pedestrian has been detected

        idx = 0

        for idx in range(len(rects)):

            print('Pedestrian detected')
            (xA, yA, xB, yB) = rects[idx]  # Get the region coord. to search for a pedestrian (corresponds to previous frame).

            # Store the corresponding keypoints and descriptors if the pedestrian has already been detected
            if len(allKeypoints) > idx:
                previousKeypoints = allKeypoints[idx]
                previousDescriptors = allDescriptors[idx]

            print('(xA, yA, xB, yB) = ', (xA, yA, xB, yB))
            pedestrian = image[yA:yB, xA:xB]  # select the region to search for the keypoints

            print('Number of previous keypoints : ', len(previousKeypoints))
            print('Number of associated descriptors : ', previousDescriptors.shape)

            # find keypoints and descriptors using SURF
            currentKeypoints, currentDescriptors = surf.detectAndCompute(pedestrian, None)
            print('Number of current keypoints found: ', len(currentKeypoints))
            print('Number of associated descriptors : ', currentDescriptors.shape)

            # update keypoints coordinates : should do it right after SURF detection to avoid issues when
            # performing brute force matching
            currentKeypoints = updateKeypointsCoordinates(currentKeypoints, xA, yA)
            print('Keypoints coordinates updated')

            # save all current keypoints & descriptors
            allCurrentKeypoints = currentKeypoints
            allCurrentDescriptors = currentDescriptors

            if DETECTION_COUNT <= idx:
                print('This is the reference frame. Not performing Brute Force Matching & LeastSquare.')
                DETECTION_COUNT += 1

            else:
                print('We should have some reference from the previous frame. Performing BFMatching & LeastSquare.')
                # perform brute force matching
                previousKeypoints, previousDescriptors, currentKeypoints, currentDescriptors = bruteForceMatching(
                    previousKeypoints, previousDescriptors, currentKeypoints, currentDescriptors)
                print('Number of matched current Keypoints : ', len(currentKeypoints))

                ########## SELECT THE FUNCTION TO FIND AN AFFINE TRANSFORM USING THE KEYPOINTS
                # 'reduced' affine transform
                #theta, alpha, tx, ty = findReducedAffTrans(previousKeypoints, currentKeypoints)

                # only a translation motion
                tx, ty = findTranslationTransf(previousKeypoints, currentKeypoints)
                theta = 0
                alpha = 1
                print('theta, alpha, tx, ty = ', theta, alpha, tx, ty)

                # 'general' affine transform
                # affTransMatrix = findGeneralAffTrans(previousKeypoints, currentKeypoints)
                #################################################################

                ########## SELECT THE CORRESPONDING FUNCTION TO UPDATE BOUNDING RECTANGLE COORDINATES
                # for 'reduced' affine transform & translation
                xA, yA, xB, yB = updateRectangleReducedAffTrans((xA, yA, xB, yB), theta, alpha, tx, ty)

                # for 'general' affine transform
                #xA, yA, xB, yB = updateRectangleGeneralAffTrans((xA, yA, xB, yB), affTransMatrix)
                ##################################################################

                # cast to int
                xA, yA, xB, yB = [int(x) for x in [xA, yA, xB, yB]]

            # save the new bounding rectangle coordinates
            rects[idx] = (xA, yA, xB, yB)

            # save all keypoints detected in this frame as the previous ones
            previousKeypoints = allCurrentKeypoints

            # save associated descriptors
            previousDescriptors = allCurrentDescriptors
            print('previousDescriptors : ', previousDescriptors.shape)

            # Replace or add previous Keypoints and descriptors of this pedestrian to the general lists of keypoints and
            #  descriptors
            if len(allKeypoints) > idx:
                allKeypoints.insert(idx,previousKeypoints)
                allKeypoints.pop(idx+1)
                allDescriptors.insert(idx, previousDescriptors)
                allDescriptors.pop(idx + 1)
            else:
                allKeypoints.append(previousKeypoints)
                allDescriptors.append(previousDescriptors)

            # draw the bounding rectangle & keypoints
            cv2.rectangle(disp_image, (xA, yA), (xB, yB), (0, 255, 0), 2)
            disp_image = cv2.drawKeypoints(disp_image, currentKeypoints, disp_image)

            PED_ALREADY_DET = True
            print('DETECTION_COUNT = ', DETECTION_COUNT)
            print('\n')

    cv2.imshow('Tracking', disp_image)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    print('\n')
