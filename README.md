# Computer Vision Fundamentals

Assignments/Projects from CSE4310 Fundamentals of Computer Vision. Details on each project and its requirements can be found within the _assignment[#].pdf_ files. NumPy and scikit-image are primarily used within the code.

## requirements.yml
requirements.yml file contains the dependencies and packages within the virtual environment used with Anaconda for each project. 
Use `conda env create -f requirements.yml` to create the environment based on that yaml file.

## CSE4310 Example References
This directory contains examples and demos from https://github.com/ajdillhoff that may be referenced throughout each project.

## Image Classification
This directory contains code for classifying images on the Food101 dataset with deep learning. The code utilizes PyTorch Lightning to create 3 different categories of networks:
- Convoutional neural network
- All convolutional neural network
- Transfer learning

There is also a version of the CNN which uses regularization in the form of dropout which can be found in _Regularization.py_

## Image Processing
The code here covers basic image processing operations implemented from scratch such as changing from color spaces RGB to HSV and vice versa. Included is also basic image transformations implemented from scratch such as random square cropping, image resizing, and color jittering.

Code for creating an image pyramid of resized copies of a user-provided image is also contained within the directory. This utilizes the code written for image transformations.

## Image Stitching
This project implements keypoint matching which is later used to stitch two images together. The image stitching involves estimating either an affine or projective transformation matrix. We compute this matrix based on keypoints extracted from the two images. We then try to refine the estimation matrix by iteratively testing the keypoint estimates with the [RANSAC algorithm](https://en.wikipedia.org/wiki/Random_sample_consensus) implemented from scratch.

## Motion Tracker
The motion tracker tracks detected objects by implementing Kalman filters with the motion detector class that supplies a list of candidate objects in motion. The Kalman filters predict the object's next step in motion and update predictions accordingly.
