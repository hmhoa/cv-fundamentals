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
    print(f'Image size: {w}x{h}')

    if sz <= 0 or sz > min(w,h)):
        sys.exit("Crop size not within range of image size")
    
    #generate random center point based on crop size
    center_w = random.randint(size, w-size-1)
    center_h = random.randint(size, h-size-1)
    print(f'Random center point chosen: ({center_w},{center_h})')

    #square crop image with random center location from which to crop
    img_crop = img[(center_w-size):(center_w+size), (center_h-size):(center_h+size)]

    return img_crop

#2. Patch extraction
def extract_patch(img, num_patches):
    return

#3. Resizes an image based on scale factor, resized using nearest neighbor interpolation
def resize_img(img, factor):
    return

#4. Randomly perturbs the HSV values on an input image by an amount no greater than the given input value
def color_jitter(img, hue, saturation, value):
    return

filename = input('Enter image name: ')
sz = int(input('Enter crop size: '))

image = io.imread(filename)
img_arr = np.asarray(image)

crop = random_crop(image, sz)
io.imsave("randomcrop.jpg", crop)
print("Random crop saved!")

patch_num = int(input('Enter the number of patches: '))
