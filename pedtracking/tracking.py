# this file is part of tdsi-project-pedestrian

import numpy as np
from scipy.spatial.distance import cdist


def bruteForceMatching(kp1, des1, kp2, des2):
    """
    This function matches SURF keypoints between 2 images using the Brute Force technique. It returns the list of
    keypoints present on the second image (i.e the next frame in a video) that have been matched with keypoints on the
    first image (i.e the previous frame in a video). The matching is done via computing the distance (in the sense of
    the L2-norm) between the respective descriptors of each keypoints. Then, the pairs of keypoints with minimal
    distance is extracted (a 'good' distance is supposed to be < 0.5) in increasing distance order.
    :param kp1: the list of keypoints determined by the SURF algorithm on the first image.
    :param des1: the numpy.ndarray of shape (len(kp1), 64) containing the descriptors associated with the first list of
    keypoints.
    :param kp2: the list of keypoints determined by the SURF algorithm on the second image.
    :param des2: the numpy.ndarray of shape (len(kp1), 64) containing the descriptors associated with the second list of
    keypoints.
    :return: the list of cv2.keypoints on the second image which have been matched with keypoints on the first image.
    """

    if (des1 is not None) & (des2 is not None):

        dists = cdist(des2, des1, p=2)  # outputs a matrix of distances of shape (len(kp2) , len(kp1) )
        # the lower the distance between 2 keypoints, the better

        copyDists = dists.copy()  # create a copy of the distances matrix

        result = []  # empty list to store the result in the form (keypoint2Idx, keypoint2Idx, distance)

        for i in range(min(len(kp1), len(kp2))):
            # get the indices of the smallest value in distances, i.e the best keypoints pair
            tempIdx = np.unravel_index(dists.argmin(), dists.shape)

            # get the associated distance
            value = dists[tempIdx[0], tempIdx[1]]

            # delete the row & column containing this minimum, as the pair of keypoints have to be independent,
            # i.e a keypoint in the first list cannot be matched to 2 or more keypoints in list 2 & vice-versa

            dists = np.delete(np.delete(dists, tempIdx[0], axis=0), tempIdx[1], axis=1)

            # get the 'original' indices of the minimum value (using an untouched copy of distances, since the indices
            # change when we remove rows or columns)
            trueIdx = np.where(copyDists == value)
            trueIdx = (trueIdx[0][0], trueIdx[1][0])  # from a tuple of np.arrays to a simpler tuple

            # append the real indices & the distance value to result
            result.append((trueIdx[0], trueIdx[1], value))

        # We consider a match between 2 keypoints to be good if the distance is < 0.5. Hence, we discard those which
        # do not respect this condition
        result = [element for element in result if element[2] < 0.8]

        # we now select the keypoints (& associated descriptors) in kp2 that were matched with a keypoint in kp1.
        keypoints = []
        descriptors = np.empty((len(result), 64))

        for idx, element in enumerate(result):
            keypoints.append(kp2[element[0]])
            descriptors[idx] = des2[element[0]]

        # finally, return the keypoints (& their descriptors) of kp2 that were matched with a keypoint in kp1, and
        # respect distance < 0.5
        return keypoints, descriptors

    else:
        print('One or both of the descriptors array is empty. Cannot perform Brute Force Matching.')
        return [], None


def updateRectangle(keypoints, delta=30):
    """
    This functions updates the bounding rectangle used to track pedestrians based on the coordinates of the keypoints
    computed on a region containing the pedestrian.
    It returns the parameters of the updated bounding rectangle.
    :param keypoints: the list of keypoints computed on the region delimited by the previous rectangle.
    :param delta: margin used to scale the rectangle on the keypoints. if 0, no margin is added. We recommend a non-null
    margin.
    :return: xA, yA, xB, yB the coordinates used to draw the rectangle afterwards.
    """
    if keypoints:
        pts = [keypoint.pt for keypoint in keypoints]

        xA = (round(min([coord[0] for coord in pts]) - delta) if round(min([coord[0] for coord in pts]) - delta) > 0 else 0)
        xB = round(max([coord[0] for coord in pts]) + delta)

        yA = (round(min([coord[1] for coord in pts]) - delta) if round(min([coord[1] for coord in pts]) - delta) > 0 else 0)
        yB = round(max([coord[1] for coord in pts]) + delta)

        return xA, yA, xB, yB
    else:
        print('The provided keypoints list is empty. (xA, yA, xB, yB) returned as null values')
        return 0, 0, 0, 0