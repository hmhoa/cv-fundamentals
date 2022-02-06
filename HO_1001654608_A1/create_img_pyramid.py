#Hoang Ho - 1001654608
#CSE 4310-001 Fundamentals of Computer Vision
#Assignment 1 - 3. Image Pyramids
#Due February 9, 2022

import numpy as np
import skimage.io as io
import sys
import os
from img_transforms import resize_img

#3. Takes in an image as numpy array as well as an integer representing  the height of the image pyramid
#creates an image pyramid of resized copies of the original image based on input pyramid height
#also saves each resized copy with the same name as the original
#fname - file name of the original image
def create_pyramid(fname, img, height):
    h, w, channels = img.shape #height, width, and channels of the image respectively

    #split the file name and file extension for later saving
    image_file = os.path.splitext(fname)

    #create list of resized imgs on the pyramid including the original image
    #each resized copy in powers of 2
    pyramid = [img]
    scale = 1
    for i in range(1,height):
        scale = scale*2
        print(scale)

        #resize_img functions takes a factor in percents, convert to percent when calling the function
        percent_factor = int((h / (scale*h))*100)
        print(percent_factor)
        resized_copy = resize_img(img, percent_factor)
        pyramid.append(resized_copy)
        io.imsave(image_file[0] + str(scale) + "x" + image_file[1], resized_copy)

    
    #combined image of resized copies including original image to display an image pyramid
    combined_img = np.zeros((h, w + w//2, channels))
    #largest/original image placed first
    combined_img[:h, :w, :] = pyramid[0]
    
    #add in resized copies
    row = 0
    #each cpy in pyramid is a resized copy image
    for cpy in  pyramid[1:]:
        cpy_row, cpy_col = cpy.shape[:2]
        combined_img[row:row+cpy_row, w:w+cpy_col, :] = cpy
        row += cpy_row

    #normalizing the values and ensuring within [0,255] range as integers
    combined_img = combined_img.astype(np.float64) / 255
    combined_img = 255 * combined_img
    combined_img = combined_img.astype(int)

    return combined_img

def main():
    filename = input('Enter image name: ')
    image = io.imread(filename)

    pyramid_height = int(input('Enter the image pyramid height: '))

    img_pyramid = create_pyramid(filename, image, pyramid_height)

    io.imshow(img_pyramid)
    io.show()

if __name__ == "__main__":
    main()