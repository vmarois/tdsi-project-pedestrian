# This file is part of tdsi-project-pedestrian

# Import lines for functions in this module

from . import detection
from . import tracking

from .detection import hogSVMDetection
from .tracking import bruteForceMatching, updateRectangle