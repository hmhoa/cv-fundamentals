# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 3 - Detecting Motion and Kalman Filters
# Due April 1, 2022 by 11:59 PM

# used following as starting point: https://github.com/ajdillhoff/CSE4310/blob/main/qtdemo.py
# references: https://github.com/ajdillhoff/CSE4310/blob/main/frame_diff.ipynb

import sys
import random
import argparse # makes it easy to write user-friendly command-line interfaces

from PySide2 import QtCore, QtWidgets, QtGui
import numpy as np
import matplotlib.pyplot as plt
from skvideo.io import vread
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import  dilation

from motion_detector import *

class QtDemo(QtWidgets.QWidget):
    def __init__(self, frames):
        super().__init__()

        self.frames = frames
        
        # create motion detector object
        self.motion_detector = MotionDetector(frames, 5, 0.05, 0.05, 3, 3)

        self.current_frame = 0

        self.next1_button = QtWidgets.QPushButton("Next Frame")
        # Add UI button to allow user to jump backward 1 frame
        self.back1_button = QtWidgets.QPushButton("Previous Frame")
        # Add UI button that allow the user to jump forward and backward by 60 frames instead of 1 frame
        self.next60_button = QtWidgets.QPushButton("Next 60 Frames")
        self.back60_button = QtWidgets.QPushButton("Previous 60 Frames")

        # Configure image label
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        h, w, c = self.frames[0].shape
        if c == 1:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))

        # Configure slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frames.shape[0]-1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.next1_button)
        # add backward 1 frame and next/back 60 frame buttons
        self.layout.addWidget(self.back1_button)
        self.layout.addWidget(self.next60_button)
        self.layout.addWidget(self.back60_button)
        self.layout.addWidget(self.frame_slider)

        # Connect functions
        self.next1_button.clicked.connect(self.on_next1_click)
        self.back1_button.clicked.connect(self.on_back1_click)
        self.next60_button.clicked.connect(self.on_next60_click)
        self.back60_button.clicked.connect(self.on_back60_click)
        self.frame_slider.sliderMoved.connect(self.on_move)

    @QtCore.Slot()
    def on_next1_click(self):
        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        self.current_frame += 1 # frame jump by 1 frame

    @QtCore.Slot()
    def on_back1_click(self):
        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        self.current_frame -= 1 # frame jump back by 1 frame

        # re-initialize
        self.motion_detector = MotionDetector(self.current_frame, 5, 0.05, 0.05, 3, 3)

    @QtCore.Slot()
    def on_next60_click(self):
        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        self.current_frame += 60 # frame jump by 60 frames

    @QtCore.Slot()
    def on_back60_click(self):
        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        self.current_frame -= 60 # frame jump back by 60 frames

        # re-initialize
        self.motion_detector = MotionDetector(self.current_frame, 5, 0.05, 0.05, 3, 3)

    @QtCore.Slot()
    def on_move(self, pos):
        self.current_frame = pos
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo for loading video with Qt5.")
    parser.add_argument("video_path", metavar='PATH_TO_VIDEO', type=str)
    parser.add_argument("--num_frames", metavar='n', type=int, default=-1)
    parser.add_argument("--grey", metavar='True/False', type=str, default=False)
    args = parser.parse_args()

    num_frames = args.num_frames

    if num_frames > 0:
        # vread returns ndarray of dimensions (T, M, N, C)
        # T: number of frames
        # M: height
        # N: width
        # C: depth
        frames = vread(args.video_path, num_frames=num_frames, as_grey=args.grey) # load video
    else:
        frames = vread(args.video_path, as_grey=args.grey)

    app = QtWidgets.QApplication([])

    widget = QtDemo(frames)
    widget.resize(800, 600)
    widget.show()   

    sys.exit(app.exec_())
