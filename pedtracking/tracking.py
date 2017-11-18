# this file is part of tdsi-project-pedestrian


import numpy as np

def bruteForceMatching(bf, kp1, des1, kp2, des2):
    """
    This functions matches SURF keypoints between 2 images using the Brute Force technique. It returns the list of
    keypoints present on the second image (i.e the next frame in a video) that have been matched with keypoints on the
    first image (i.e the previous frame in a video).
    :param bf: the BruteForceMatcher object
    :param kp1: the list of keypoints determined by the SURF algorithm on the first image.
    :param des1: the numpy.ndarray of shape (len(kp1), 64) containing the descriptors associated with the first list of
    keypoints.
    :param kp2: the list of keypoints determined by the SURF algorithm on the second image.
    :param des2: the numpy.ndarray of shape (len(kp1), 64) containing the descriptors associated with the second list of
    keypoints.
    :return: the list of cv2.keypoints on the second image which have been matched with keypoints on the first image.
    """

    # get the list of matches of keypoints between kp1 & kp2
    matches = bf.match(des1, des2)

    # we now select the keypoints in kp2 which have been matched with keypoints in kp1
    keypointsList = []

    for match in matches:
        keypointsList.append(kp2[match.trainIdx])  # the trainIdx param of cv2.DMatch is the index of the descriptor
        # in the train descriptors list, i.e the list of descriptors associated with the second image.

    # we select the associated descriptors
    descriptors = np.ndarray((len(keypointsList), 64))

    for idx, match in enumerate(matches):
        descriptors[idx] = des2[match.trainIdx]

    return keypointsList, descriptors


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

    pts = [keypoint.pt for keypoint in keypoints]

    xA = (round(min([coord[0] for coord in pts]) - delta) if round(min([coord[0] for coord in pts]) - delta) > 0 else 0)
    xB = round(max([coord[0] for coord in pts]) + delta)

    yA = (round(min([coord[1] for coord in pts]) - delta) if round(min([coord[1] for coord in pts]) - delta) > 0 else 0)
    yB = round(max([coord[1] for coord in pts]) + delta)

    return xA, yA, xB, yB