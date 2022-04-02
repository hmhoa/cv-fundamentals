# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 3 - Detecting Motion and Kalman Filters
# Due April 6, 2022 by 11:59 PM

# references: https://github.com/ajdillhoff/CSE4310/blob/main/frame_diff.ipynb
#             https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label
#             https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

import sys

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import skvideo.io
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import  dilation

# label - check connecting pixels and assign them same label so easy tracking (anything moving are nonzero pixels)
# dilation - grow the blobs

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
        self.last_frame_index = 0 # last frame detected recently

        # maintain list of object candidates detected over time
        # each element is [region, kalman filter object, last frame updated]
        self.motion_objs = []
        # at least 3 frames needed for initialization
        initial_objs = self.detect(3)
        
        # append each kalman filter as part of the initialization
        for obj in initial_objs:
            pos = obj.centroid
            state = np.array([[pos[0], pos[1], 1, 1]]) # state representation for kalman filter
            self.motion_objs.append([obj, KalmanFilter(state.T),3])
            
            # if maximum objects detected, break
            if obj.label == self.max_objs:
                break

    # detect all the different objects in motion
    # at least 3 frames needed for initialization
    # returns potential candidates/regions of blobs where an object in motion was detected
    def detect(self, index):
        ppframe = rgb2gray(self.src_frames[index-2]) # load in previous previous frame
        pframe = rgb2gray(self.src_frames[index-1]) # load in previous frame
        cframe = rgb2gray(self.src_frames[index]) # load in current frame

        # simplest way to detect motion - frame differencing
        diff1 = np.abs(cframe - pframe)
        diff2 = np.abs(pframe - ppframe)

        motion_frame = np.minimum(diff1, diff2) # anything moving is a nonzero pixel
        thresh_frame = motion_frame > self.motion_t # filter out small pixel noise based on threshold set
        dilated_frame = dilation(thresh_frame, np.ones((9,9))) # grow sparse pixels that aren't fully connected
        label_frame = label(dilated_frame) # give each detected motion object a label

        return regionprops(label_frame) # gives bounding box for each label detected (region of blobs detected) - give back centroids of groups of objects

    def update_tracking(self, new_frame_index):
        # check if current frame is past the number of skipped frames since last detection
        # detect only if it has been [skips] frames since last detection
        i = self.last_frame_index
        while(i < new_frame_index):
            regions = self.detect(i)
            
            # check if each centroid coordinate for each region/blob object found belongs to an object currently being tracked using the distance threshold
            # if not add it to the list of objects to track if there are not already max objs in the list
            for candidate in regions:
                cpos = candidate.centroid # centroid coordinate tuple of current candidate
                measurement = (np.array([[cpos[0], cpos[1], 1, 1]])).T # measurement for kalman filter

                # check if candidate belongs to an object currently being tracked based on distance threshold
                # If the distance between an object proposal and the prediction of one of the filters is
                # less than δ (dist_t), assume that proposal is a measurement for the corresponding filter and update # predictions for the filter.
                dist_diff = []
                for prediction in self.motion_objs:
                    prediction[1].predict(self.skips)
                    model = prediction[1].state_model
                    distance = sqrt(np.sum(np.square(model-measurement)))
                    dist_diff.append(distance)
                matches = [ match for match in dist_diff if match < self.dist_t ]
            
                # update the prediction
                if len(matches) != 0:
                    # get index of matched filter
                    match_index = dist_diff.index(min(matches))
                    self.motion_objs[match_index][1].update(measurement)
                    self.motion_objs[match_index][2] = i
                    print(f'Updated prediction for motion object {self.motion_objs[match_index][0].label}')
                else: # add kalman filter
                    if len(self.motion_objs) < self.max_objs:
                        print(f'Added motion object {candidate.label} to tracked objects')
                        self.motion_objs.append([candidate, KalmanFilter(measurement), i])

            i += self.skips # skip number of frames between each detection
            
        self.last_frame_index = new_frame_index
        # remove inactive objects using frame_hyst
        j = 0
        while(j < len(self.motion_objs)):
            last_updated = self.motion_objs[j][2]
            if new_frame_index-last_updated >= self.frame_hyst:
                self.motion_objs.pop(j)
                print(f'Removed motion object {self.motion_objs[j][0].label} from tracked objects')
            j += 1
