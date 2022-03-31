# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 3 - Detecting Motion and Kalman Filters
# Due March 30, 2022 by 11:59 PM

import numpy as np
import sys
from kalman_filter import *

class MotionDetector:
    # frame_hyst: α - Frame hysteresis for determining active or inactive objects
    # motion_t: τ - The motion threshold for filtering out noise
    # dist_t: δ - A distance threshold to determine if an object candidate belongs to an object
    #   currently being tracked
    # frames: s - The number of frames to skip between detections. The tracker will still work
    #   well even if it is not updated every frame
    # max_objs: N - The number of maximum objects to detect
    def __init__(self, frame_hyst, motion_t, dist_t, frames, max_objs):
        self.frame_hyst = 0
        self.motion_t = 0
        self.dist_t = 0
        self.frames = 0
        self.max_objs = 0
