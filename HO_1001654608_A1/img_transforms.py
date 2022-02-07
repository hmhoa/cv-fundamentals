#Hoang Ho - 1001654608
#CSE 4310-001 Fundamentals of Computer Vision
#Assignment 1 - 2. Image Transformations
#Due February 9, 2022

import numpy as np
from numpy.lib import stride_tricks
import skimage.io as io
from skimage import img_as_ubyte
import sys
import random

#for hsv function in other code
from change_hsv import *

#1. Generate random square crop of an image
def random_crop(img, size):
    #get width and length of img
    w = np.size(img, axis=1) #columns
    h = np.size(img, axis=0) #rows
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

#2. Patch extraction, returns n^2 non-overlapping patches given an input image as a numpy array as well as an integer n that is the total number of patches made
#Assumed input image is a square
def extract_patch(img, num_patches):
    #non-overlapping patches of size n
    h, w = img.shape[:2] #unpack height and width
    n = num_patches*num_patches
    shape = [h // num_patches, w // num_patches] + [num_patches,num_patches]
    print(shape)

    #(row, col, patch_row, patch_col)
    #:2 grabs all the elements up to not including index 2
    strides = [num_patches * s for s in img.strides[:2]] + list(img.strides[:2])
    print(strides)
    #extract patches
    #specify shape and strides to define hwo tor traverse the array for viewing (as patches in this case)
    patches = stride_tricks.as_strided(img, shape=shape, strides=strides)

    return patches, n

#3. Resizes an image based on scale factor, resized using nearest neighbor interpolation
#assuming number representing the scale factor is greater than 0
def resize_img(img, factor):
    orig_w = np.size(img, axis=1)
    orig_h = np.size(img, axis=0)
    new_w = (int)(orig_w * factor)
    new_h = (int)(orig_h * factor)

    print(f'Original image size: {orig_w}x{orig_h}')
    print(f'Resized image size: {new_w}x{new_h}')
    print(f'Scale factor: {factor}')
    
    resized_img = np.zeros((new_h, new_w, 3))

    for i in range(new_h):
        for j in range(new_w):
            mapping_x = (int)(i/factor)
            mapping_y = (int)(j/factor)
            resized_img[i,j,:] = img[mapping_x,mapping_y,:]

    #normalizing the values and ensuring within [0,255] range as integers
    resized_img = resized_img.astype(np.float64) / 255
    resized_img = 255 * resized_img
    resized_img = resized_img.astype(int)

    return resized_img

#4. Randomly perturbs the HSV values on an input image by an amount no greater than the given input value
#Assuming hue, saturation, and value given are positive values
def color_jitter(img, hue, saturation, value):
    w = np.size(img, axis=1)
    h = np.size(img, axis=0)
    jittered_img = RGBtoHSV(img)

    #validate inputs
    if hue < 0 or hue > 360:
        sys.exit("Hue input is not within 0 to 360 degrees.")
    if saturation < 0 or saturation > 1:
        sys.exit("Saturation input is not within 0 to 1")
    if value < 0 or value > 1:
        sys.exit("Value input is not within 0 to 1")

    #randomly select value no greater than the given input value range from hue, saturation, and value
    rand_h = random.randint((-1*hue), hue)
    rand_s = random.uniform((-1*saturation), saturation)
    rand_v = random.uniform((-1*value), value)

    #do hsv modifications
    jittered_img[:,:,0] += rand_h
    jittered_img[:,:,1] += rand_s
    jittered_img[:,:,2] += rand_v

    #modified values exceed range for HSV values, so cap them at the max allowed
    jittered_img[jittered_img[:,:,0] > 360] = 360
    jittered_img[jittered_img[:,:,1] > 1] = 1
    jittered_img[jittered_img[:,:,2] > 1] = 1

    #modified values fall below range for HSV values, so cap them at the min allowed
    jittered_img[jittered_img[:,:,:] < 0] = 0

    #turn back into RGB
    jittered_img = HSVtoRGB(jittered_img)

    return jittered_img

#testing outputs
def main():
    
    #using same first image for all tests
    filename = input('Enter image name: ')
    image = io.imread(filename)

    sz = int(input('Enter crop size: '))
    crop = random_crop(image, sz)
    io.imshow(crop)
    io.show()

    patch_num = int(input('Enter the number of patches: '))
    patched, n = extract_patch(image, patch_num)
    print(f'Total number of patches: {n}')

    scale = float(input('Enter resize scale factor (in percent %): '))
    resized = resize_img(image, scale)
    io.imshow(resized)
    io.show()

    h = int(input('Enter the hue: '))
    s = float(input('Enter the saturation: '))
    v = float(input('Enter the value: '))
    jittered = color_jitter(image, h, s, v)
    io.imshow(jittered)
    io.show()

if __name__ == "__main__":
    main()