# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 3 - Detecting Motion and Kalman Filters
# Due April 1, 2022 by 11:59 PM

import sys

import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import  dilation

# label - check connecting pixels and assign them same label so easy tracking (anything moving are nonzero pixels)
# dilation - 

from kalman_filter import *

class MotionDetector:
    # frame_hyst: α - Frame hysteresis for determining active or inactive objects
    # motion_t: τ - The motion threshold for filtering out noise
    # dist_t: δ - A distance threshold to determine if an object candidate belongs to an object
    #   currently being tracked
    # skips: s - The number of frames to skip between detections. The tracker will still work
    #   well even if it is not updated every frame
    # max_objs: N - The number of maximum objects to detect
    def __init__(self, src_frames, frame_hyst, motion_t, dist_t, skips, max_objs):
        self.src_frames = src_frames
        self.frame_hyst = frame_hyst
        self.motion_t = motion_t
        self.dist_t = dist_t
        self.skips = skips
        self.max_objs = max_objs

        # maintain list of object candidates detected over time
        self.motion_objs = []
        # at least 3 frames needed for initialization
        self.update_tracking(3)

    def update_tracking(self, index):
        # detect all the different objects
        # at least 3 frames needed for initialization
        ppframe = rgb2gray(self.src_frames[index-2]) # load in previous previous frame
        pframe = rgb2gray(self.src_frames[index-1]) # load in previous frame
        cframe = rgb2gray(self.src_frames[index]) # load in current frame
        diff1 = np.abs(cframe - pframe)
        diff2 = np.abs(pframe - ppframe)

        motion_frame = np.minimum(diff1, diff2) # anything moving is a nonzero pixel
        thresh_frame = motion_frame > self.motion_t # filter out small pixel noise based on threshold set
        dilated_frame = dilation(thresh_frame, np.ones((9,9))) # grow sparse pixels that aren't fully connected
        label_frame = label(dilated_frame) # give back centroids of groups of objects
        regions = regionprops(label_frame) # gives bounding box for each label detected (region of blobs detected)

#TODO: determine how to separate different objects detected from each frame
#      and connect it to the gui somehow with skips
#      all video frames are kept in the gui? does the motion detector look at only 3 frames at a time
