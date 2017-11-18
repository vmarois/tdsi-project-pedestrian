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