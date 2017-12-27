# this file is part of tdsi-project-pedestrian
# THIS FILE SHOULD CONTAIN ALL OBJECT TRACKING FUNCTIONS USING LEAST SQUARE METHODS

import numpy as np
import math
import cv2


def findSeparatedAffTrans(previousKpts, currentKpts):
    """
    This function uses the coordinates of the keypoints detected in the current & previous frames to compute the
    scaling & translation parameters (along the x- & y-axis) defining the affine transformation which explains the
    motion of the rectangle bounding the pedestrian.
    We define this affine transformation as follow : let (x2,y2) be the coordinates of a keypoint in the current frame
    and (x1, y1) its coordinates in the previous frame. Then:
    x2 = tx + sx.x1
    y2 = ty + xy.y1
    Remark: motion along x-axis does not interfere with motion along y-axis (i.e 'separated' in the function name)
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


def updateRectangleSeparatedAffTrans(rectCoord, scaling, translation):
    """
    Updates the bounding rectangle based on the affine transformation returned by findSeparatedAffTrans().
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


def findReducedAffTrans(previousKpts, currentKpts):
    """
    This function uses the coordinates of the keypoints detected in the current & previous frames to compute the
    rotation, scaling & translation parameters (along the x- & y-axis) defining the affine transformation which explains
    the motion of the rectangle bounding the pedestrian.
    This function assumes the affine transform is as following :
    [x; y] -> [a -b; b a] * [x; y] + [tx; ty]
    [a -b; b a] corresponds to a rotation matrix, with rotation angle given by atan(b/a) and scaling parameter
    sqrt(a**2 + b**2).
    Remark: As we impose the shape of a rotation matrix, we are reducing the number of affine degrees
    of freedom (i.e 'reduced' in the function name).
    :param previousKpts: the list of keypoints detected in the previous frame.
    :param currentKpts: the list of keypoints detected in the current frame.
    :return: rotation_angle, scaling, tx, ty
    """

    # select only 4 best matches
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

        # solution of the form x = [a, b, tx, ty]' = ((A' * A)^-1) * A' * b
        x = np.linalg.inv(A.T * A) * A.T * b

        # compute each parameter
        theta = math.atan2(x[1, 0], x[0, 0])  # outputs angle in [-pi, pi]
        alpha = math.sqrt(x[0, 0] ** 2 + x[1, 0] ** 2)
        tx = x[2, 0]
        ty = x[3, 0]

        return theta, alpha, tx, ty

    else:
        print('One or both of the keypoints lists is empty. Cannot perform least square regression.')
        return 0, 0, 0, 0


def updateRectangleReducedAffTrans(rectCoord, theta, alpha, tx, ty):
    """
    Updates the bounding rectangle based on the affine transformation returned by findReducedAffineTransform().
    :param rectCoord: (xA, yA, xB, yB) the current keypoints coordinates
    :param theta: the rotation parameter
    :param alpha: the scaling parameter along both axis
    :param tx: the translation parameter along x-axis
    :param ty: the translation parameter along y-axis
    :return: xA, yA, xB, yB updated.
    """

    # get current rectangle coordinates
    xA, yA, xB, yB = rectCoord

    # define the 4 bounding points
    rect_pts = np.array([[[xA, yA]], [[xB, yA]], [[xA, yB]], [[xB, yB]]], dtype=np.float32)

    # warp the affine transform into a full perspective transform
    affine_warp = np.array([[alpha * np.cos(theta), -alpha * np.sin(theta), tx],
                            [alpha * np.sin(theta), alpha * np.cos(theta), ty],
                            [0, 0, 1]], dtype=np.float32)

    new_rect_pts = cv2.perspectiveTransform(rect_pts, affine_warp)
    new_xA = new_rect_pts[0, 0, 0]
    new_yA = new_rect_pts[0, 0, 1]
    new_xB = new_rect_pts[3, 0, 0]
    new_yB = new_rect_pts[3, 0, 1]

    return new_xA, new_yA, new_xB, new_yB


def findGeneralAffTrans(previousKpts, currentKpts):
    """
    This function uses the coordinates of the keypoints detected in the current & previous frames to compute the
    rotation, scaling & translation parameters (along the x- & y-axis) defining the affine transformation which explains
    the motion of the rectangle bounding the pedestrian.
    This function assumes the affine transform is as following :
    [x; y] -> [a b; c d] * [x; y] + [tx; ty]
    [a b; c d] represents the affine rotation, scale, and stretch.
    Remark: As we do not impose any constraint on a,b,c,d, we are in the general case of the affine transform
    (i.e 'general' in the function name).
    :param previousKpts: the list of keypoints detected in the previous frame.
    :param currentKpts: the list of keypoints detected in the current frame.
    :return: the affine transform matrix [a b; c d; tx ty]
    """
    # select 4 best matches only
    previousKpts = previousKpts[:5]
    currentKpts = currentKpts[:5]

    if (previousKpts is not None) & (currentKpts is not None):
        # build A matrix of shape [Nb of keypoints, 3]

        A = np.ndarray(((len(previousKpts), 3)))

        for idx, keypoint in enumerate(previousKpts):
            # Keypoint.pt = (x-coord, y-coord)
            A[idx, :] = [keypoint.pt[0], keypoint.pt[1], 1]

        # build b matrix of shape [Nb of keypoints, 2]
        b = np.ndarray((len(previousKpts), 2))

        for idx, keypoint in enumerate(currentKpts):
            b[idx, :] = [keypoint.pt[0], keypoint.pt[1]]

        # convert the numpy.ndarrays to matrix :
        A = np.matrix(A)
        b = np.matrix(b)

        # solution of the form x = [x1, x2, x3, x4]' = ((A' * A)^-1) * A' * b
        x = np.linalg.inv(A.T * A) * A.T * b

        return x

    else:
        print('One or both of the keypoints lists is empty. Cannot find affine transform.')
        return 0


def updateRectangleGeneralAffTrans(rectCoord, x):
    """
    Updates the bounding rectangle based on the affine transformation returned by findGeneralAffineTransform().
    :param rectCoord: (xA, yA, xB, yB) the current keypoints coordinates
    :param x: the affine transform matrix [a b; c d; tx ty]
    :return: xA, yA, xB, yB updated.
    """
    # get current rectangle coordinates
    xA, yA, xB, yB = rectCoord

    # define the 4 bounding points
    rect_pts = np.array([[[xA, yA]], [[xB, yA]], [[xA, yB]], [[xB, yB]]], dtype=np.float32)

    # warp the affine transform into a full perspective transform
    affine_warp = np.array([[x[0, 0], x[1, 0], x[2, 0]],
                            [x[0, 1], x[1, 1], x[2, 1]],
                            [0, 0, 1]], dtype=np.float32)

    new_rect_pts = cv2.perspectiveTransform(rect_pts, affine_warp)
    new_xA = new_rect_pts[0, 0, 0]
    new_yA = new_rect_pts[0, 0, 1]
    new_xB = new_rect_pts[3, 0, 0]
    new_yB = new_rect_pts[3, 0, 1]

    return new_xA, new_yA, new_xB, new_yB