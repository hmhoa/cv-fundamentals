# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 2 - SIFT and RANSAC
# Due March 11, 2022 by 11:59 PM

# ensure latest version of scikit-image with SIFT installed
# conda  search scikit-image -c conda-forge
# conda install scikit-image=0.19.2 -c conda-forge

#do imports
from math import sqrt
from sys import float_info
import numpy as np
import PIL.Image as Image #enable reading images
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from matplotlib.patches import ConnectionPatch
from skimage.feature import match_descriptors, plot_matches, SIFT
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize, ProjectiveTransform, SimilarityTransform, warp
from skimage import measure



# 2 Keypoint Matching
# keypoint1, keypoint2 - list of keypoint coordinates from dst and src images respectively
# keypoints represented as a (N,2) array. Keypoint coordinates as (row, col)
# returns a list of indices of matching feature pairs where a pair (i,j) in the list indicates a match between the feature at index i in the src image with the feature at index j in the second image
def match(keypoints1, keypoints2):
    #list of tuple pairs (i,j) of matching keypoints
    matches = []
    
    # choose the keypoint array with the smallest length to determine for loop iteration amount
    # keypoints1_size = np.size(keypoints1, axis=0)
    # keypoints2_size = np.size(keypoints2, axis=0)
    # length = min(keypoints1_size, keypoints2_size)

    # using brute force and calculating the lowest matches using euclidean distance
    # outer loop iterates through each keypoint in keypoints1
    # kp = keypoint, so kp1 is a keypoint from keypoints 1 and kp2 is a keypoint from keypoints 2
    i = 0
    for kp1 in keypoints1:
        #reset indices counter and smallest distance for next keypoint to analyze
        j = 0
        smallest_distance = float_info.max
        
        # iterate through each keypoint in keypoints2 and find the smallest euclidean distance of kp1 and kp2
        for kp2 in keypoints2:
            distance = sqrt(((kp2[0]-kp1[0])**2)+((kp2[1]-kp1[1])**2)) #calculate euclidean distance of the current keypoint pair
            if distance < smallest_distance:
                smallest_distance = distance
                #update current indices of matching feature pair
                matching_pair = (i,j)
            j += 1
        #append this matching_pair to the matches list if it is not already in the list
        if matching_pair not in matches:
            matches.append(matching_pair)

        #increment to track which keypoint/index we are on for keypoints1
        i += 1

    return matches
    


# 2.1 Plot Keypoint Matches
# plotting function that combines two input images of the same size side-by-side and plots the keypoitns detected in each  image. Plot the lines  between the keypoints shwoing which ones have been matched together
# calculate the size of the images that you're given (both will be same resolution), so double it


# 3 Image Stitching
# 3.1 Estimate Affine MAtrix
# 3.2 Estimate Projective Matrix
# 3.3 RANSAC
# follow basic approach of randomly selecting poitns fitting the  model that are goon  points, taking entire dataset and determining if it was a good point dedpending on number of inliers
#destination image is the one not being warped

# 3.4 Testing
def main():
    # set up images
    
    dst_img_rgb = np.asarray(Image.open('a2_images/Rainier1.png'))
    src_img_rgb = np.asarray(Image.open('a2_images/Rainier2.png'))

    # checking if theres a 4th channel - alpha and turning back into rgb
    if dst_img_rgb.shape[2] == 4:
        dst_img_rgb = rgba2rgb(dst_img_rgb)
    if src_img_rgb.shape[2] == 4:
        src_img_rgb = rgba2rgb(src_img_rgb)

    # grayscale the images
    dst_img = rgb2gray(dst_img_rgb)
    src_img = rgb2gray(src_img_rgb)


    # plotting the set up images (nothing changed so far)
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img, cmap='gray')
    ax2.imshow(src_img, cmap='gray')

    # 1 Keypoint Detection - extract features that describe the object in the image
    # descriptor - describe/define features despite orientation, size, position, etc.
    # can do harris corner detection
    dst_detector = SIFT()
    dst_detector.detect_and_extract(dst_img) #detects keypoints using FAST corner detection and extracts rBRIEF descriptors
    dst_keypoints = dst_detector.keypoints #represented as a (N,2) array. Keypoint coordinates as (row, col)
    dst_descriptors = dst_detector.descriptors #represented as a (N, n_hist*n_hist*n_ori) array; default is n_hist = 4, n_ori = 8
    
    src_detector = SIFT()
    src_detector.detect_and_extract(src_img)
    src_keypoints = src_detector.keypoints
    src_descriptors = src_detector.descriptors

    print(f'Destination Keypoints ({np.size(dst_keypoints, axis=0)} keypoints):')
    print(dst_keypoints)
    print(f'\nSource Keypoints ({np.size(src_keypoints, axis=0)}):')
    print(src_keypoints)

    # 2 Keypoint Matching
    matches = match(dst_keypoints, src_keypoints)
    print(f'\n{len(matches)} Matches found (i,j):')
    print(matches)

    


if __name__ == "__main__":
    main()