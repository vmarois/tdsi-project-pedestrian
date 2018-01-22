# TDSI project on pedestrian detection & tracking in video #

A `Python` project for the detection & tracking of pedestrians in a video. The proposed framework is mainly relying on [openCV 3](https://docs.opencv.org/3.1.0/d1/dfb/intro.html).
The project is developed by :

* Vincent Marois <vincent.marois@protonmail.com>
* Théophile Dupre <theophile.dupre@insa-lyon.fr>
* Maud Neyrat <maud.neyrat@insa-lyon.fr>

## Description

The main objective of this project is to detect pedestrian and follow them in a sequence of 500 images, rendering a video at ~15 fps. Several constraints are shaping up this problem :

* Multiple pedestrians are visible on the sequence of images,
* Some of these pedestrians are crossing each other,
* The point of view of the camera implies that the size of the pedestrians is changing throughout the video,
* The implemented algorithm needs to run in near real time (15 fps ).

## Retained solution

We are separating the problem in 2 sub-problems :
* Pedestrian detection
* Object tracking

For the first part, we chose a combination of the histogram of oriented gradients (hog) as a descriptor and a SVM as a classifier able to recognize if a pedestrian is present in the selected region of interest. The advantage of this technique is that openCV comes with a pre-trained SVM, making the implementation easier.

For the object tracking part, we are working on a [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)-based algorithm :
* The pedestrian is delimited by a bounding rectangle which needs to be updated at each frame.
* Using the coordinates of the bounding rectangle, we can compute SIFT keypoints within the region of the frame corresponding to the bounding rectangle.
* Then, brute force matching is performed to only select current keypoints that were present in the previous frame (selection based on descriptors l2-norm & spatial distance).
* The _tracking_ part can be done via several methods :
* We can compute the coordinates of the center of mass of the matched keypoints detected in the ROI containining the pedestrian and center the bounding rectangle using these coordinates. This technique presents the weakness of not using the previous frame as a reference. We also need to reduce the size of the bounding rectangle to deal with the depth of field.
* We can use the pairs of matched keypoints to estimate the affine transform (using a least square resolution) (rotation, scaling, translation in (x,y)) fitting the motion of the pedestrian and use it to update the corner points of the rectangle. This approach is a **_work in progress_** for now, but we hope it will be more robust than the first one.

### How do I get set up? ###
* Make sure OpenCV is installed,
* Just clone the repository :


        git clone git@gitlab.in2p3.fr:olivier.bernard/tdsi-project-pedestrian.git
    

### Who do I talk to? ###

The project administrators are:

* Olivier Bernard <olivier.bernard@creatis.insa-lyon.fr>

### How is structured this project ? ###

We are structuring the project by creating a local `Python` module  called  **pedtracking**, which contains the methods we implemented as we need them. So far, this module is structured in 3  `Python` files, and each file contains some methods relevant with the topic indicated by the  `Python` file name. Hence:
* In  `least_square_tracking.py`, you'll find functions based on least square methods to estimate several affine transforms using pairs of keypoints, and update the bounding rectangle.
* In `tracking.py`, you'll find functions related to the first tracking approach, e.g. centering the bounding rectangle on the center of mass of the current keypoints. The brute force matching function is also in this file.
* In `detection.py`, available functions wrap the {HOG + SVM} approach (with some non-max suppression), and a test of background substraction.

This structure allows :
- For clearer and shorter main scripts,
- We are able to gather some functions by their role in the project,
- Code maintenance and the addition of future features are simplified.
