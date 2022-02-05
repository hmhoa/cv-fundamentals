#Hoang Ho - 1001654608
#CSE 4310-001 Fundamentals of Computer Vision
#Assignment 1 - 2. Image Transformations
#Due February 9, 2022

import numpy as np
import skimage.io as io
import skimage.color as color
import sys
import random

#1. Generate random square crop of an image
def random_crop(img, size):
    #get width and length of img
    w = len(img[0])
    h = len(img)
    
    #generate random center point based on crop size
    center_w = random.randint(size, w-size-1)
    center_h = random.randint(size, h-size-1)

    #square crop image with random center location from which to crop
    img_crop = img[(center_w-size):(center_w+size), (center_h-size):(center_h+size)]

    return img_crop

#2. Patch extraction
def extract_patch(img, num_patches):

#3. Resizes an image
def resize_img(img, factor):

#4. Randomly perturbs the HSV values on an input image by an amount no greater than the given input value
def color_jitter(img, hue, saturation, value):

filename = input('Enter image name: ')
sz = input('Enter crop size: ')

image = io.imread(filename)
img_arr = np.asarray(image)

if sz > 0 and sz <= min(len(img_arr), len(img_arr[0])):
    crop = random_crop(image, sz)
    io.imsave("randomcrop.jpg", crop)
else:
    sys.exit("Crop size not within range of image size")
