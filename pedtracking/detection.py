# this file is part of tdsi-project-pedestrian
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


def hogSVMDetection(hog, image):
    """
    This functions detects pedestrians in the passed image using the hog signature and the pre-trained SVM classifier
    available in cv2. It returns the bounding rectangles for each pedestrian detected in the image.
    :param hog: the HOG descriptor
    :param image: the input image
    :return: a numpy.ndarray containing the (x-coordinate, y-coordinate, x-coordinate + width, y-coordinate + height)
    for each rectangle.
    """
    rects, _ = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.03)
    # the 'scale'  is one of the most important parameters to tune when performing pedestrian detection

    # apply non-maxima suppression to the bounding boxes using a fairly large overlap threshold to try to maintain
    # overlapping boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    return rects


def backgroundSubstraction(backgroundsubstractor, image):
    """
    This function separates the foreground from the background of the passed image using a built-in openCV function.
    It also applies a morphologic operation (open then close) to reduce the noise.
    :param backgroundsubstractor: the cv2.bgsegm.createBackgroundSubtractorMOG() instanciation.
    :param image: the image to execute background separation on.
    :param kernel: the structuring element used for morphologic filtering.
    :return: the foreground image
    """

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 30))


    fgmask = np.uint8(backgroundsubstractor.apply(image)/255)

    # filter the bushes on the mask (maybe not necessary)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel1, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel2, iterations=2)

    # apply the mask on the image
    image[:, :, 0] = image[:, :, 0] * fgmask
    image[:, :, 1] = image[:, :, 1] * fgmask
    image[:, :, 2] = image[:, :, 2] * fgmask

    return image


def removeRedundantRectangles(rects):
    rect_middle = []

    for rect in rects:
        rect_middle = np.append(rect_middle, [(rect[2] + rect[0]) / 2, (rect[3] + rect[1]) / 2], axis=0)

    i = 0
    rect_idx = []

    while i < len(rect_middle):
        k = 0
        while k < len(rect_middle):
            if k != i:
                if np.bool(abs(rect_middle[i] - rect_middle[k]) < 10) & np.bool(abs(rect_middle[i + 1] - rect_middle[k + 1]) < 10):
                    f = np.sort((i / 2, k / 2))
                    rect_idx = np.append(rect_idx, f, axis=0)
            k += 2
        i += 2

    rect_idx = [elt for idx, elt in enumerate(rect_idx) if idx % 2 != 0]
    rect_idx = set(rect_idx)
    rect_idx = sorted(rect_idx, reverse=True)
    for value in rect_idx:
        rects = np.delete(rects, value, axis=0)

    return rects
