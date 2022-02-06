#Hoang Ho - 1001654608
#CSE 4310-001 Fundamentals of Computer Vision
#Assignment 1 - 2. Image Transformations
#Due February 9, 2022

import numpy as np
import skimage.io as io
import skimage.color as color
import skimage.transform as transform
from skimage import img_as_ubyte
import sys
import random

#1. Generate random square crop of an image
def random_crop(img, size):
    #get width and length of img
    w = np.size(img, axis=1)
    h = np.size(img, axis=0)
    print(f'Image size: {w}x{h}')

    if sz <= 0 or sz > min(w,h):
        sys.exit("Crop size not within range of image size")
    
    #generate random center point based on crop size
    center_w = random.randint(size, w-size-1)
    center_h = random.randint(size, h-size-1)
    print(f'Random center point chosen: ({center_w},{center_h})')

    #square crop image with random center location from which to crop
    img_crop = img[(center_h-size):(center_h+size), (center_w-size):(center_w+size)]

    return img_crop

#2. Patch extraction
def extract_patch(img, num_patches):
    return

#3. Resizes an image based on scale factor, resized using nearest neighbor interpolation
#assuming integer representing the scale factor is greater than 0 and is a percentage %
def resize_img(img, factor):
    orig_w = np.size(img, axis=1)
    orig_h = np.size(img, axis=0)
    scale = factor / 100
    new_w = (int)(orig_w * scale)
    new_h = (int)(orig_h * scale)

    print(f'Original image size: {orig_w}x{orig_h}')
    print(f'Resized image size: {new_w}x{new_h}')
    print(f'Scale factor: {scale}')
    
    resized_img = np.zeros((new_h, new_w, 3))

    for i in range(new_h):
        for j in range(new_w):
            mapping_x = (int)(i/scale)
            mapping_y = (int)(j/scale)
            resized_img[i,j,:] = img[mapping_x,mapping_y,:]

    #normalizing the values and ensuring within [0,255] range as integers
    resized_img = resized_img.astype(np.float64) / 255
    resized_img = 255 * resized_img
    resized_img = resized_img.astype(int)

    return resized_img

#4. Randomly perturbs the HSV values on an input image by an amount no greater than the given input value
def color_jitter(img, hue, saturation, value):
    return

filename = input('Enter image name: ')
image = io.imread(filename)


sz = int(input('Enter crop size: '))
crop = random_crop(image, sz)
io.imshow(crop)
io.show()

patch_num = int(input('Enter the number of patches: '))
patched = extract_patch(image, patch_num)

scale = int(input('Enter resize scale factor (in percent %): '))
resized = resize_img(image, scale)
io.imshow(resized)
io.show()