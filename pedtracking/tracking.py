# this file is part of tdsi-project-pedestrian

import numpy as np
import math
from scipy.spatial.distance import cdist
import cv2

def bruteForceMatching(kp1, des1, kp2, des2):
    """
    This function matches SURF keypoints between 2 images using the Brute Force technique.
    It returns the list of keypoints (& their descriptors) present on the second image (i.e the next frame in a video)
    that have been matched with keypoints on the first image (i.e the previous frame in a video).
    It also returns the list of keypoints (& their descriptors) on the first image matched that have been matched in
    the second image.
    The matching is done via computing the distance (in the sense of the L2-norm) between the respective descriptors
    of each keypoints.
    Then, the pairs of keypoints with minimal distance is extracted (a 'good' distance is supposed to be < 0.5)
    in increasing distance order.
    :param kp1: the list of keypoints determined by the SURF algorithm on the first image.
    :param des1: the numpy.ndarray of shape (len(kp1), 64) containing the descriptors associated with the first list of
    keypoints.
    :param kp2: the list of keypoints determined by the SURF algorithm on the second image.
    :param des2: the numpy.ndarray of shape (len(kp2), 64) containing the descriptors associated with the second list of
    keypoints.
    :return: The keypoints & their descriptors of both images which have been matched with a distance < 0.5
    """

    xA = 0
    yA = 0

    if (des1 is not None) & (des2 is not None):

        dists = cdist(des2, des1, metric='minkowski', p=2)  # outputs a matrix of distances of shape (len(kp2) , len(kp1) )
        # the lower the distance between 2 keypoints, the better
        copyDists = dists.copy()  # create a copy of the distances matrix

        result = []  # empty list to store the result in the form (keypoint2Idx, keypoint1Idx, distance)

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
        result = [element for element in result if element[2] < 0.5]

        # we now select the keypoints (& associated descriptors) in kp2 that were matched with a keypoint in kp1.
        nextkeypoints = []
        nextdescriptors = np.empty((len(result), 64))

        # We also select the matched keypoints in the previous frame (used to update the bounding rectangle
        # via a least square method). The order of selection is important!
        prevkeypoints = []
        prevdescriptors = np.empty((len(result), 64))

        for idx, element in enumerate(result):
            nextkeypoints.append(kp2[element[0]])
            nextdescriptors[idx] = des2[element[0]]
            prevkeypoints.append((kp1[element[1]]))
            prevdescriptors[idx] = des1[element[1]]


        # Delete matched keypoints separated by more than 10 pixels
        i = 0
        while i < len(prevkeypoints):
            if( np.sqrt(np.square(nextkeypoints[i].pt[0] - prevkeypoints[i].pt[0]) + np.square(nextkeypoints[i].pt[1] - prevkeypoints[i].pt[1])) > 10):
                prevkeypoints = np.delete(prevkeypoints, i)
                prevdescriptors = np.delete(prevdescriptors, i, axis=0)
                nextkeypoints = np.delete(nextkeypoints, i)
                nextdescriptors = np.delete(nextdescriptors, i, axis=0)
            else:
                i += 1

        return prevkeypoints, prevdescriptors, nextkeypoints, nextdescriptors

    else:
        print('One or both of the descriptors array is empty. Cannot perform Brute Force Matching.')
        return [], None, [], None


def updateRectangle(keypoints, xMargin=30, yMargin=30):
    """
    This functions updates the bounding rectangle used to track pedestrians based on the coordinates of the keypoints
    computed on a region containing the pedestrian.
    It returns the parameters of the updated bounding rectangle.
    :param keypoints: the list of keypoints computed on the region delimited by the previous rectangle.
    :param xMargin: margin used to scale the rectangle on the x axis. if 0, no margin is added. We recommend a non-null
    margin.
    :param yMargin: margin used to scale the rectangle on the y axis. if 0, no margin is added. We recommend a non-null
    margin (probably higher than the xMargin)
    :return: xA, yA, xB, yB the coordinates used to draw the rectangle afterwards.
    """
    if keypoints:
        pts = [keypoint.pt for keypoint in keypoints]

        xA = (round(min([coord[0] for coord in pts]) - xMargin) if round(min([coord[0] for coord in pts]) - xMargin) > 0 else 0)
        xB = round(max([coord[0] for coord in pts]) + xMargin)

        yA = (round(min([coord[1] for coord in pts]) - yMargin) if round(min([coord[1] for coord in pts]) - yMargin) > 0 else 0)
        yB = round(max([coord[1] for coord in pts]) + yMargin)

        return xA, yA, xB, yB
    else:
        print('The provided keypoints list is empty. (xA, yA, xB, yB) returned as null values')
        return 0, 0, 0, 0


def updateMargin(xMarginStart, yMarginStart, NbTotImg, countFromDetection):
    """
    This function updates the margin used to modify the bounding rectangle size.
    Since the size of the pedestrian is decreasing with every new frame, so should the bounding rectangle size.
    The method used here is a linear decrease : we start to decrease the margin when the pedestrian has been detected,
    at a rate of (NbTotImg - countFromDetection)/NbTotImg, with countFromDetection increasing at each new frame.
    It returns the updated x-axis margin & y-axis margin
    :param xMarginStart: The initial margin for the x axis.
    :param yMarginStart: The initial margin for the y axis.
    :param NbTotImg: The total number of frames in the video
    :param countFromDetection: the index of the current, taking for reference the frame where the pedestrian has been
    detected.
    :return: xMargin, yMargin updated.
    """

    xMargin = xMarginStart
    yMargin = yMarginStart

    # define the decreasing rate
    rate = (NbTotImg - countFromDetection)/NbTotImg

    if countFromDetection/NbTotImg < 0.6:
        xMargin = round(xMarginStart * rate)
        yMargin = round(yMarginStart * rate)

    return xMargin, yMargin


def updateKeypointsCoordinates(keypoints, xA, yA):
    """
    This function updates the keypoints coordinates, setting the origin to (xA, yA).
    As we compute the SIFT keypoints on the rectangle bounding the pedestrian, their coordinates have for reference
    the rectangle and not the original image. This function fixes that, by setting their reference back to the original
    image
    :param keypoints: the list of keypoints to update
    :param xA: the x-coordinate used for reference
    :param yA: the y-coordinate used for reference
    :return: the list of keypoints with updated coordinates.
    """
    for keypoint in keypoints:
        (x, y) = keypoint.pt
        x += xA
        y += yA
        keypoint.pt = (x, y)
    return keypoints


def updateRectangleCenter(keypoints, xMargin=30, yMargin=30):
    """
    New function test to update bounding rectangle : instead of using exterior keypoints as deliminitors, compute
    center of mass of keypoints and center bounding rectangle on it.
    :param keypoints: the list of keypoints computed on the region delimited by the previous rectangle.
    :param xMargin: margin used to scale the rectangle on the x axis. if 0, no margin is added. We recommend a non-null
    margin.
    :param yMargin: margin used to scale the rectangle on the y axis. if 0, no margin is added. We recommend a non-null
    margin (probably higher than the xMargin)
    :return: xA, yA, xB, yB new coordinates of bounding rectangle
    """
    if keypoints:
        xcoord = [keypoint.pt[0] for keypoint in keypoints]
        ycoord = [keypoint.pt[1] for keypoint in keypoints]
        xCenter = int(np.mean(xcoord))
        yCenter = int(np.mean(ycoord))

        xA = xCenter - xMargin
        xB = xCenter + xMargin
        yA = yCenter - yMargin
        yB = yCenter + yMargin


        return xA, yA, xB, yB

    else:
        print('The provided keypoints list is empty. (xA, yA, xB, yB) returned as null values')
        return 0, 0, 0, 0


def leastSquareRegression(previousKpts, currentKpts):
    """
    This function uses the coordinates of the keypoints detected in the current & previous frames to compute the
    scaling & translation parameters (along the x- & y-axis) defining the affine transformation which explains the
    motion of the rectangle bounding the pedestrian.
    We define this affine transformation as follow : let (x2,y2) be the coordinates of a keypoint in the current frame
    and (x1, y1) its coordinates in the previous frame. Then:
    x2 = tx + sx.x1
    y2 = ty + xy.y1
    Note : we suppose the keypoints coordinates have the same origin.
    :param previousKpts: the list of keypoints detected in the previous frame.
    :param currentKpts: the list of keypoints detected in the current frame.
    :return: sx, sy, tx, ty
    """
    if (previousKpts is not None) & (currentKpts is not None):
        # we're using the same notation as in the lecture
        # Keypoint.pt = (x-coord, y-coord)

        # for the x-coordinates problem
        Xx = np.ones((len(previousKpts), 2))
        Yx = np.zeros((len(currentKpts), 1))

        # for the x-coordinates problem
        Xy = np.ones((len(previousKpts), 2))
        Yy = np.zeros((len(currentKpts), 1))

        # we now collect the x- & y-coordinates of the current keypoints:
        for idx, keypoint in enumerate(currentKpts):
            Yx[idx] = keypoint.pt[0]
            Yy[idx] = keypoint.pt[1]

        # do the same for the previous keypoints:
        for idx, keypoint in enumerate(previousKpts):
            Xx[idx, 1] = keypoint.pt[0]
            Xy[idx, 1] = keypoint.pt[1]

        # convert the numpy.ndarrays to matrix :
        Xx = np.matrix(Xx)
        Xy = np.matrix(Xy)
        Yx = np.matrix(Yx)
        Yy = np.matrix(Yy)

        # solution of the form A = [t,s]' = ((X' * X)^-1) * X' * Y
        Ax = np.linalg.inv(Xx.T * Xx) * Xx.T * Yx
        print(Ax)
        tx, sx = np.asscalar(Ax[0][0]), np.asscalar(Ax[1][0])

        Ay = np.linalg.inv(Xy.T * Xy) * Xy.T * Yy
        print(Ay)
        ty, sy = np.asscalar(Ay[0][0]), np.asscalar(Ay[1][0])

        return sx, sy, tx, ty

    else:
        print('One or both of the keypoints lists is empty. Cannot perform least square regression.')
        return 0, 0, 0, 0


def updateRectangleLeastSquare(rectCoord, scaling, translation):
    """
    Updates the bounding rectangle based on the affine transformation found when performing least square regression
    on the keypoints coordinates.
    :param rectCoord: (xA, yA, xB, yB) the current keypoints coordinates
    :param scaling: (sx, sy) the scaling parameters along x- & y-axis
    :param translation: (tx, ty) the translation parameters along x- & y-axis
    :return: xA, yA, xB, yB updated.
    """

    # get current rectangle coordinates
    xA, yA, xB, yB = rectCoord

    # get scaling parameters
    sx, sy = scaling

    # get translation parameters
    tx, ty = translation

    xA = sx * xA + tx
    xB = sx * xB + tx

    yA = sy * yA + ty
    yB = sy * yB + ty

    if xA < 0:
        xA = 0
    if yA < 0:
        yA = 0

    return xA, yA, xB, yB


def leastSquareRegression2D(previousKpts, currentKpts):
    """
    This function uses the coordinates of the keypoints detected in the current & previous frames to compute the
    rotation, scaling & translation parameters (along the x- & y-axis) defining the affine transformation which explains
    the motion of the rectangle bounding the pedestrian.
    We are using T.Grenier's code as the base theory for the transformation.
    Note : we suppose the keypoints coordinates have the same origin.
    :param previousKpts: the list of keypoints detected in the previous frame.
    :param currentKpts: the list of keypoints detected in the current frame.
    :return: rotation_angle, scaling, tx, ty
    """

    previousKpts = previousKpts[:5]
    currentKpts = currentKpts[:5]

    if (previousKpts is not None) & (currentKpts is not None):
        # build A matrix of shape [2 * Nb of keypoints, 4]
        A = np.ndarray(((2 * len(previousKpts), 4)))

        for idx, keypoint in enumerate(previousKpts):
            # Keypoint.pt = (x-coord, y-coord)
            A[2 * idx, :] = [keypoint.pt[0], -keypoint.pt[1], 1, 0]
            A[2 * idx + 1, :] = [keypoint.pt[1], keypoint.pt[0], 0, 1]

        # build b matrix of shape [2 * Nb of keypoints, 1]
        b = np.ndarray((2 * len(previousKpts), 1))

        for idx, keypoint in enumerate(currentKpts):
            b[2 * idx, :] = keypoint.pt[0]
            b[2 * idx + 1, :] = keypoint.pt[1]

        # convert the numpy.ndarrays to matrix :
        A = np.matrix(A)
        b = np.matrix(b)

        # solution of the form x = [x1, x2, x3, x4]' = ((A' * A)^-1) * A' * b
        x = np.linalg.inv(A.T * A) * A.T * b

        theta = math.atan2(x[1, 0], x[0, 0])  # outputs angle in [-pi, pi]
        alpha = math.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)
        bx = x[2, 0]
        by = x[3, 0]


        return theta, alpha, bx, by

    else:
        print('One or both of the keypoints lists is empty. Cannot perform least square regression.')
        return 0, 0, 0, 0


def updateRectangleLeastSquare2D(rectCoord, theta, alpha, tx, ty):
    """
    Updates the bounding rectangle based on the affine transformation found when performing least square regression
    on the keypoints coordinates.
    :param rectCoord: (xA, yA, xB, yB) the current keypoints coordinates
    :param theta: the rotation parameter
    :param alpha: the scaling parameter along both axis
    :param tx: the translation parameter along x-axis
    :param ty: the translation parameter along y-axis
    :return: xA, yA, xB, yB updated.
    """

    # get current rectangle coordinates
    xA, yA, xB, yB = rectCoord

    # define rotation matrix from the theta angle
    rotmatrix = np.asmatrix(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]))
    newCoordA = np.asmatrix(np.array([[xA], [yA]]))
    newCoordB = np.asmatrix(np.array([[xB], [yB]]))

    #newCoordA = alpha * rotmatrix * newCoordA + np.asmatrix(np.array([[tx], [ty]]))
    #newCoordB = alpha * rotmatrix * newCoordB + np.asmatrix(np.array([[tx], [ty]]))

    # without rotation
    newCoordA = alpha * newCoordA + np.asmatrix(np.array([[tx], [ty]]))
    newCoordB = alpha * newCoordB + np.asmatrix(np.array([[tx], [ty]]))

    if newCoordA[0, 0] < 0:
        newCoordA[0, 0] = 0
    if newCoordA[1, 0] < 0:
        newCoordA[1, 0] = 0

    return newCoordA[0, 0], newCoordA[1, 0], newCoordB[0, 0], newCoordB[1, 0]


def homographyMatrix(previousKpts, currentKpts):
    src_pts = np.float32([kp.pt for kp in previousKpts]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp.pt for kp in currentKpts]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    M = np.asmatrix(M)
    print('Homography matrix M : \n', M)
    return M


def updateRectangleHomography(rectCoord, M):
    # get current rectangle coordinates
    xA, yA, xB, yB = rectCoord

    newCoordA = np.asmatrix(np.array([[xA], [yA], [1]]))
    newCoordB = np.asmatrix(np.array([[xB], [yB], [1]]))

    newCoordA = M * newCoordA
    newCoordB = M * newCoordB

    return (newCoordA[0,0]/newCoordA[2,0]), (newCoordA[1,0]/newCoordA[2,0]), (newCoordB[0,0]/newCoordB[2,0]), (newCoordB[1,0]/newCoordB[2,0])

def findTranslationTransf(previousKpts, currentKpts):
    """
    This function finds the translation between the previous and current keypoints
    :param previousKpts: the list of keypoints detected in the previous frame.
    :param currentKpts: the list of keypoints detected in the current frame.
    :return: amount of translation along the x and y-axis
    """

    tx, ty = 0,0
    for i, kp in enumerate(previousKpts):
        tx += currentKpts[i].pt[0] - previousKpts[i].pt[0]
        ty += currentKpts[i].pt[1] - previousKpts[i].pt[1]

    return tx/len(previousKpts), ty/len(previousKpts)