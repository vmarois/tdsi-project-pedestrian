# This file is part of tdsi-project-pedestrian

# Import lines for functions in this module

from . import detection
from . import tracking

from .detection import hogSVMDetection, backgroundSubstraction
from .tracking import bruteForceMatching, updateRectangle, updateMargin, updateKeypointsCoordinates, \
    updateRectangleCenter, leastSquareRegression, updateRectangleLeastSquare, leastSquareRegression2D, \
    updateRectangleLeastSquare2D, findTranslationTransf
