#Hoang Ho - 1001654608
#CSE 4310-001 Fundamentals of Computer Vision
#Assignment 2 - SIFT and RANSAC
#Due March 11, 2022 by 11:59 PM

#ensure latest version of scikit-image with SIFT installed
#conda  search scikit-image -c conda-forge
#conda install scikit-image=0.19.2 -c conda-forge

#do imports
import numpy as np
import PIL.Image as Image #enable reading images
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from matplotlib.patches import ConnectionPatch
from skimage.feature import match_descriptors, plot_matches, SIFT
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize, ProjectiveTransform, SimilarityTransform, warp
from skimage import measure



#2 Keypoint Matching
#returns a list of indices of matching feature pairs
#pair (i,j) in the list indicates a match between the feature at index i in the src image with the feature at index j in the second image
def match(keypoints1, keypoints2):
    return
    #lowest match w L2 distance


#2.1 Plot Keypoint Matches
#calculate the size of the images that you're given (both will be same resolution), so double it


#3 Image Stitching
#3.1 Estimate Affine MAtrix
#3.2 Estimate Projective Matrix
#3.3 RANSAC
#follow basic approach of randomly selecting poitns fitting the  model that are goon  points, taking entire dataset and determining if it was a good point dedpending on number of inliers
#destination image is the one not being warped

#3.4 Testing
def main():
    #set up images
    
    dst_img_rgb = np.asarray(Image.open('a2_images/Rainier1.png'))
    src_img_rgb = np.asarray(Image.open('a2_images/Rainier2.png'))

    #checking if theres a 4th channel - alpha and turning back into rgb
    if dst_img_rgb.shape[2] = 4:
        dst_img_rgb = rgba2rgb(dst_img_rgb)
    if src_img_rgb.shape[2] = 4:
        src_img_rgb = rgba2rgb(src_img_rgb)

    #grayscale the images
    dst_img = rgb2gray(dst_img_rgb)
    src_img = rgb2gray(src_img_rgb)


    #plotting the set up images (nothing changed so far)
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img, cmap='gray')
    ax2.imshow(src_img, cmap='gray')

    #1 Keypoint Detection - extract features that describe the object in the image
    #descriptor - describe/define features despite orientation, size, position, etc.
    #can do harris corner detection
    dst_detector = SIFT()
    dst_detector.detect_and_extract(dst_img) #detects keypoints using FAST corner detection and extracts rBRIEF descriptors
    dst_keypoints = dst_detector.keypoints #represented as a (N,2) array. Keypoint coordinates as (row, col)
    dst_descriptors = dst_detector.descriptors #represented as a (N, n_hist*n_hist*n_ori) array; default is n_hist = 4, n_ori = 8
    
    src_detector = SIFT()
    src_detector.detect_and_extract(src_img)
    src_keypoints = src_detector.keypoints
    src_descriptors = src_detector.descriptors

    #2 Keypoint Matching


if __name__ == "__main__":
    main()