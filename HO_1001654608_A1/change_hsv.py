#Hoang Ho - 1001654608
#CSE 4310-001 Fundamentals of Computer Vision
#Assignment 1 - 1. Color Spaces
#Due February 9, 2022

from multiprocessing.sharedctypes import Value
import numpy as np
import skimage.io as io
import skimage.color as color

#1.1 RGB to HSV
#convert rgb image to hsv
def RGBtoHSV(image):
    
    hsv_img = image[:,:,:]

    #iterate through each pixel in the image
    for x in range(len(hsv_img)):
        for y in range(len(hsv_img[x])):
            #grab rgb values from each channel of the current pixel
            r = hsv_img[x][y][0]
            g = hsv_img[x][y][1]
            b = hsv_img[x][y][2]
            
            #calculate value
            v = np.max(hsv_img, axis=2)

            #compute chroma
            c = v - np.min(hsv_img, axis=2)
            
            #compute saturation
            if v = 0:
                s = 0
            else:
                s = c/v
            
            #calculate hue
            if c = 0:
                h_prime = 0
            elif v = r:
                h_prime = ((g-b)/c)%6
            elif v = g:
                h_prime = ((b-r)/c)+2
            elif v = b:
                h_prime = ((r-b)/c)+4
            h = 60 * h_prime

            #change to hsv values
            hsv_img[x][y][0] = h
            hsv_img[x][y][1] = s
            hsv_img[x][y][2] = v

    return hsv_img

#1.2 HSV to RGB
#convert hsv image to rgb
def HSVtoRGB(image):
    rgb_img = image[:,:,:]

    #iterate through each pixel in the image
    for x in range(len(rgb_img)):
        for y in range(len(rgb_img[x])):
            #grab hsv values from each channel of the current pixel
            h = rgb_img[x][y][0]
            s = rgb_img[x][y][1]
            v = rgb_img[x][y][2]

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
            elif h_prime >= 5 and h_prime < 6:
                rgb_prime = (c,0,x)
            
            #calculate final rgb value
            m = v - c
            r = rgb_prime[0]+m
            g = rgb_prime[1]+m
            b = rgb_prime[2]+m

            #change to rgb values
            rgb_img[x][y][0] = r
            rgb_img[x][y][1] = g
            rgb_img[x][y][2] = b
    
    return rgb_img


def main(args):
    num_args = len(args)

    #check if there are arguments
    if num_args < 4:
        sys.exit("Not enough inputs made, inputs for the image file name, hue, saturation, and value needed.")

    file_name = args[0]
    #assuming the first argument contains the file name of the image
    img = io.imread(file_name) #img represented as a numpy array
    io.imshow(img)

    #assuming inputs are integer numbers
    hue = int(args[1])
    saturation = int(args[2])
    value = int(args[3])

    #validate inputs
    if hue < 0 or hue > 360:
        sys.exit("Hue input is not within 0 to 360 degrees.")
    if saturation < 0 or saturation > 1:
        sys.exit("Saturation input is not within 0 to 1")
    if value < 0 or value > 1:
        sys.exit("Value input is not within 0 to 1")
    
    #convert to hsv
    hsv_img = RGBtoHSV(img)

    #do hsv modifications
    hsv_img[:,:,0] += hue
    hsv_img[:,:,1] += saturation
    hsv_img[:,:,2] += value

    #convert back to rgb
    new_rgb_img = HSVtoRGB(hsv_img)

    #save modified image
    io.imsave(file_name+"_modified.jpg", new_rgb_img)


if __name__ == "__main__":
    main(sys.argv)