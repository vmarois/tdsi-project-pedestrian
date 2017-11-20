# TDSI project on pedestrian detection & tracking in video #

A `Python` project for the detection & tracking of pedestrians in a video. The proposed framework is mainly relying on [openCV 3](https://docs.opencv.org/3.1.0/d1/dfb/intro.html).
The project is developed by :

* Vincent Marois <vincent.marois@insa-lyon.fr>
* Théophile Dupre <theophile.dupre@insa-lyon.fr>
* Maud Neyrat <maud.neyrat@insa-lyon.fr>

## Description

The main objective of this project is to detect pedestrian and follow them in a sequence of 500 images, rendering a video at ~15 fps. Several constraints are shaping up this problem :

* Multiple pedestrians are vsisible on the sequence of images,
* Some of these pedestrians are crossing each other,
* The point of view of the camera implies that the size of the pedestrians is changing throughout the video,
* The implemented algorithm needs to run in near real time (15 fps ).

## Retained solution

We are separating the problem in 2 sub-problems for now :
* Pedestrian detection
* Object tracking

For the first part, we chose a combination of the histogram of oriented gradients (hog) as a descriptor and a SVM as a classifier able to recognize if a pedestrian is present in the selected region of interest. The advantage of this technique is that openCV comes with a pre-trained SVM, making the implementation easier.
For the object tracking part, we are working on a [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)-based algorithm. Some of our initial ideas are to perform a SIFT feature extraction on a region centered around the pre-detected pedestrian and brute-force matching with the next frame.

### How do I get set up? ###

* Just clone the repository :


git clone git@gitlab.in2p3.fr:olivier.bernard/tdsi-project-pedestrian.git

### Who do I talk to? ###

The project administrators are:

* Olivier Bernard <olivier.bernard@creatis.insa-lyon.fr>


