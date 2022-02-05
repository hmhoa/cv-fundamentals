#Hoang Ho - 1001654608
#CSE 4310-001 Fundamentals of Computer Vision
#Assignment 1 - 1. Color Spaces
#Due February 9, 2022

import numpy as np
import skimage.io as io
import skimage.color as color
from skimage import img_as_ubyte
import sys

#1.1 RGB to HSV
#convert rgb image to hsv
def RGBtoHSV(image):
    
    row = len(image)
    col = len(image[0])
    hsv_img = np.zeros((row,col,3))

    #iterate through each pixel in the image
    for i in range(row):
        for j in range(col):
            #grab rgb values from each channel of the current pixel and normalize them to be in range [0,1]
            r = image[i][j][0] / 255
            g = image[i][j][1] / 255
            b = image[i][j][2] / 255

            #calculate value
            v = np.max(image[i][j])

            #compute chroma
            c = v - np.min(image[i][j])
            
            #compute saturation
            if v == 0:
                s = 0.0
            else:
                s = c/v
            
            #calculate hue
            if c == 0:
                h_prime = 0
            elif v == r:
                h_prime = ((g-b)/c)%6
            elif v == g:
                h_prime = ((b-r)/c)+2
            elif v == b:
                h_prime = ((r-b)/c)+4
            h = 60 * h_prime

            #change to hsv values
            hsv_img[i][j][0] = h
            hsv_img[i][j][1] = s
            hsv_img[i][j][2] = v

    return hsv_img

#1.2 HSV to RGB
#convert hsv image to rgb
def HSVtoRGB(image):
    row = len(image)
    col = len(image[0])
    rgb_img = np.zeros((row,col,3))

    #iterate through each pixel in the image
    for i in range(row):
        for j in range(col):
            #grab hsv values from each channel of the current pixel
            h = image[i][j][0]
            s = image[i][j][1]
            v = image[i][j][2]

            #calculate chroma value
            c = v * s

            h_prime = h/60
            x = c * (1 - abs(h_prime%2 - 1))

            #calculate (R', G', B')
            if h_prime >= 0 and h_prime < 1:
                rgb_prime = (c,x,0)
            elif h_prime >= 1 and h_prime < 2:
                rgb_prime = (x,c,0)
            elif h_prime >= 2 and h_prime < 3:
                rgb_prime = (0,c,x)
            elif h_prime >= 3 and h_prime < 4:
                rgb_prime = (0,x,c)
            elif h_prime >= 4 and h_prime < 5:
                rgb_prime = (x,0,c)
            elif h_prime >= 5 and h_prime <= 6:
                rgb_prime = (c,0,x)
            
            #calculate final rgb value
            m = v - c
            r = (rgb_prime[0]+m)*255
            g = (rgb_prime[1]+m)*255
            b = (rgb_prime[2]+m)*255

            #change to rgb values
            rgb_img[i][j][0] = int(r)
            rgb_img[i][j][1] = int(g)
            rgb_img[i][j][2] = int(b)
    
    return rgb_img


def main(args):
    num_args = len(args)

    #check if there are arguments
    if num_args < 5:
        sys.exit("Not enough inputs made, inputs for the image file name, hue, saturation, and value needed.")

    #print out input
    #first argument should be the python file name
    print(args)

    file_name = args[1]
    img = io.imread(file_name) #img represented as a numpy array
    io.imshow(img)

    #assuming inputs are integer numbers
    hue = int(args[2])
    saturation = float(args[3])
    val = float(args[4])

    #validate inputs
    if hue < 0 or hue > 360:
        sys.exit("Hue input is not within 0 to 360 degrees.")
    if saturation < 0 or saturation > 1:
        sys.exit("Saturation input is not within 0 to 1")
    if val < 0 or val > 1:
        sys.exit("Value input is not within 0 to 1")
    
    #convert to hsv
    print("Converting to HSV")
    hsv_img = RGBtoHSV(img)

    #do hsv modifications
    print("Doing HSV modifications")
    if hue > 0:
        hsv_img[:,:,0] += hue
    if saturation > 0:
        hsv_img[:,:,1] += saturation
    if val > 0:
        hsv_img[:,:,2] += val

    #convert back to rgb
    print("Converting to RGB")
    new_rgb_img = img_as_ubyte(HSVtoRGB(hsv_img))

    #save modified image
    io.imsave("modified_"+file_name, new_rgb_img)
    print("Modified Image Saved!")

if __name__ == "__main__":
    main(sys.argv)