# Hoang Ho - 1001654608
# CSE 4310-001 Fundamentals of Computer Vision
# Assignment 2 - SIFT and RANSAC
# Due March 11, 2022 by 11:59 PM

# references https://github.com/ajdillhoff/CSE4310/blob/main/ransac.ipynb
#            https://dillhoffaj.utasites.cloud/posts/scale_invariant_feature_transforms/
#            https://dillhoffaj.utasites.cloud/posts/random_sample_consensus/
#            https://dillhoffaj.utasites.cloud/posts/image_features/
#            https://en.wikipedia.org/wiki/Random_sample_consensus

# ensure latest version of scikit-image with SIFT installed
# conda  search scikit-image -c conda-forge
# conda install scikit-image=0.19.2 -c conda-forge

# do imports
from asyncio.windows_events import NULL
from math import sqrt
from sys import float_info

from pyparsing import null_debug_action
import numpy as np
import PIL.Image as Image #enable reading images
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
# from matplotlib.patches import ConnectionPatch
from skimage.feature import SIFT
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize, ProjectiveTransform, SimilarityTransform, warp
from skimage import measure

# 2 Keypoint Matching
# dst_features, src_features - set of keypoint features from dst and src images respectively
# keypoints represented as a (N,2) array. Keypoint coordinates as (row, col)
# keypoint feature descriptors are represented as a (N, n_hist*n_hist*n_ori) array
# returns a list of indices of matching feature pairs where a pair (i,j) in the list indicates a match between the feature at index i in the src image with the feature at index j in the second image
def match(dst_features, src_features):
    print('\nFinding matches...')
    #list of pairs [i,j] of matching keypoints
    matches = []

    dst_descriptors = dst_features.descriptors #vector of all values of histograms
    src_descriptors = src_features.descriptors

    # using brute force and calculating the lowest matches using euclidean distance
    # outer loop iterates through each keypoint in keypoints1
    # d = descriptor, so d1 is a descriptor from dst_descriptors and d2 is a descriptor from src_descriptors
    i = 0
    for d1 in dst_descriptors:
        #reset indices counter and smallest distance for next keypoint to analyze
        j = 0
        smallest_distance = float_info.max
        
        # iterate through each keypoint in keypoints2 and find the smallest euclidean distance of d1 and d2
        for d2 in src_descriptors:
            squared_differences = np.square((d2[:]-d1[:])) #subtracts values for each dimension and squares it
            distance = sqrt(np.sum(squared_differences)) #calculate euclidean distance of the current descriptor pair
            if distance < smallest_distance:
                smallest_distance = distance
                #update current indices of matching feature pair
                matching_pair = [i,j]
            j += 1
        #append this matching_pair to the matches list if it is not already in the list
        if matching_pair not in matches:
            matches.append(matching_pair)

        #increment to track which keypoint/index we are on for keypoints1
        i += 1

    return matches
    
# 2.1 Plot Keypoint Matches
# plotting function that combines two input images of the same size side-by-side and plots the keypoint detected in each  image. Plot the lines  between the keypoints shwoing which ones have been matched together
# calculate the size of the images that you're given (both will be same resolution), so double it
# dst_image, src_image - input images to be combined side by side as np arrays
# dst_kp_matches, src_kp_matches - keypoints that match with destination image and source image
def plot_matches(dst_image, src_image, dst_kp_matches, src_kp_matches):
    print('\nPlotting matches...')
    # combine the two images into one image
    height, width, channels = dst_image.shape

    total_width = width*2

    combined_img = np.zeros((height, total_width,channels))
    print(f'\nCombined Image Shape: {combined_img.shape}')
    combined_img[:height, :width, :] = dst_image
    combined_img[:height, width:, :] = src_image

    # plot the points and connect them
    num_matches = len(dst_kp_matches)

    for i in range(num_matches):
        # retrieve the points
        dst_point = (dst_kp_matches[i,1], dst_kp_matches[i,0])
        src_point = (src_kp_matches[i,1], src_kp_matches[i,0])

        #pass a list of two matching keypoints we want to connect to plt.plot
        plt.plot([dst_point[0], src_point[0]+width], [dst_point[1],src_point[1]], 'ro-') # ro- specifies we want to plot red marker with a line connecting the two points

    #to get rid of the warning: Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    element_type = str(combined_img[0].dtype) # get a string that says what type each element in the image is (float or int)
    if 'float' in element_type: # is the array elements' type considered a float?
        combined_img = combined_img/np.amax(combined_img) # amax returns maximum along an array
    elif 'int' in element_type:
        combined_img = np.array(combined_img/np.amax(combined_img)*255, np.uint8)

    plt.imshow(combined_img)
    plt.show()


# 3 Image Stitching
# 3.1 Estimate Affine Matrix
# takes in set of points from source image and their matching points in the destination image
# using these samples, the affine transformation matrix is computed using normal equations
# returns a 3 x 3 matrix
def compute_affine_transform(dst_points, src_points):
    # utilize the least squares approach
    # construct the x matrix that is made up of the matching points from the source image
    src_sz = len(src_points)*2 # number of rows needed for all the source image points
    src_matrix = np.zeros((src_sz,6))
    row = 0 # track row for the matrix
    for point in src_points:
        x = point[0] # x coordinate from source image
        y = point[1] # y coordinate from source image
        src_matrix[row,0] = x
        src_matrix[row,1] = y
        src_matrix[row,2] = 1
        src_matrix[row+1,3] = x
        src_matrix[row+1,4] = y
        src_matrix[row+1,5] = 1
        row += 2

    # construct the x-hat / b matrix that is made up of the matching points from the destination image
    dst_sz = len(dst_points)*2 # number of rows needed for all the destination image points - this should be the same as src_sz
    dst_matrix = np.zeros((dst_sz,1))
    row = 0 # reset row count to use for the destination image
    for point in dst_points:
        x = point[0] # x coordinate from destination image
        y = point[1] # y coordinate from destination image
        dst_matrix[row,0] = x
        dst_matrix[row+1,0] = y
        row += 2

    # computes the vector x that approximately solves the equation a @ x = b
    # this is a least-squares solution to the system ax=b where
    # x = transform_matrix
    # a = src_matrix
    # b = dst_matrix
    # this should return a tuple containing the solution, residuals (the sum), rank (matrix rank of input a), and singular values of input a
    # solution, residuals, rank, singular_vals = np.linalg.lstsq(src_matrix, dst_matrix)
    
    # alternative method - solved analytically using normal equations
    src_inv = np.linalg.pinv(src_matrix.T @ src_matrix)
    solution = src_inv @ src_matrix.T @ dst_matrix

    # FOR TESTING: print(f'\nSolution matrix:\n {solution}')

    # construct the affine transformation matrix
    # the third row 0 0 1 indicates it is an affine transformation
    affine_transform_mtx = np.array([[solution[0,0], solution[1,0], solution[2,0]]
                                    ,[solution[3,0], solution[4,0], solution[5,0]]
                                    ,[0, 0, 1]])

    print("\nAffine transformation matrix found: ")
    print(affine_transform_mtx)

    return affine_transform_mtx

# 3.2 Estimate Projective Matrix
# takes in set of points from source image and their matching points in the destination image
# using these samples, the projective transformation matrix is computed using normal equations
# returns a 3 x 3 matrix
def compute_projective_transform(dst_points, src_points):
    # utilize the least squares approach
    # construct the x matrix that is made up of the matching points from the source image
    src_sz = len(src_points)*2 # number of rows needed for all the source image points
    src_matrix = np.zeros((src_sz,8))
    row = 0 # track row for the matrix
    for point in src_points:
        x = point[0] # x coordinate from source image
        y = point[1] # y coordinate from source image

        xd = point[0] # x coordinate from destination image
        yd = point[1] # y coordinate from destination image

        # 1st row set
        src_matrix[row,0] = x
        src_matrix[row,1] = y
        src_matrix[row,2] = 1
        src_matrix[row,6] = -x*xd
        src_matrix[row,7] = -y*xd
        
        # 2nd row set
        src_matrix[row+1,3] = x
        src_matrix[row+1,4] = y
        src_matrix[row+1,5] = 1
        src_matrix[row+1,6] = -x*yd
        src_matrix[row+1,7] = -y*yd
        
        row += 2

    # construct the x-hat / b matrix that is made up of the matching points from the destination image
    dst_sz = len(dst_points)*2 # number of rows needed for all the destination image points - this should be the same as src_sz
    dst_matrix = np.zeros((dst_sz,1))
    row = 0 # reset row count to use for the destination image
    for point in dst_points:
        x = point[0] # x coordinate from destination image
        y = point[1] # y coordinate from destination image
        dst_matrix[row,0] = x
        dst_matrix[row+1,0] = y
        row += 2

    # computes the vector x that approximately solves the equation a @ x = b
    # this is a least-squares solution to the system ax=b where
    # x = transform_matrix
    # a = src_matrix
    # b = dst_matrix
    # this should return a tuple containing the solution, residuals (the sum), rank (matrix rank of input a), and singular values of input a
    # solution, residuals, rank, singular_vals = np.linalg.lstsq(src_matrix, dst_matrix)
    
    # alternative method - solved analytically using normal equations
    src_inv = np.linalg.pinv(src_matrix.T @ src_matrix)
    solution = src_inv @ src_matrix.T @ dst_matrix

    # FOR TESTING: print(f'\nSolution matrix:\n {solution}')

    # construct the affine transformation matrix
    # the third row 0 0 1 indicates it is an affine transformation
    projective_transform_mtx = np.array([[solution[0,0], solution[1,0], solution[2,0]]
                                    ,[solution[3,0], solution[4,0], solution[5,0]]
                                    ,[solution[6,0], solution[7,0], 1]])

    print("\nProjective transformation matrix found: ")
    print(projective_transform_mtx)

    return projective_transform_mtx

# 3.3 RANSAC
# ransac frequently used for real world sensor data - helps identify outliers
# trial and error approach that groups inlier and outlier sets
#
# randomly draw 2 data points and fit a line through them (treat these as inliers) > check how many of remaining data points aside from sample points that agree with the line (inliers) as a score > repeat process set amount of times > select model with highest score as solution
#
# follow basic approach of randomly selecting points fitting the  model that are good  points, taking entire dataset and determining if it was a good point depending on number of inliers
# destination image is the one not being warped
# 
# params:
# dst_keypoints, src_keypoints - set of keypoints in destination image and their potential matches in the source image; this is the data/set of observations*
# tf_model - whose parameters is used to transform the input with yields a close approximation to the targets
# iterations - number of iterations for RANSAC algorithm to run; maximum number of iterations allowed in the algorithm
# min_samples - minimum number of samples to fit a  model with; minimum number of data points required to estimate model parameters
# threshold_boundary - a threshold boundary; a threshold value to determine data points that are fit well by model; represents the maximum acceptable distance/pixel distance the approximation can be from the model. Since input points are usually in terms of pixels, a threshold of 1 works
# d - number of close data points required to assert that a model fits well to data
# 
# returns:
# model parameters which best fit the data (or null if no good model is found)
# best_fit - the best transform matrix / model
# best_inliers - the best inlier matches as a tuple of (best dst inlier matches, best src inlier matches)
def ransac(dst_keypoints, src_keypoints, tf_model, iterations, min_samples, threshold_boundary, d):
    i = 0
    best_fit = NULL
    best_err = float_info.max # something really large
    best_inliers = (dst_keypoints, src_keypoints)

    while i < iterations:
        # step 1: randomly sample matched keypoints
        rows = src_keypoints.shape[0]
        samples_output_shape = (min_samples)
        samples_indices = np.random.randint(0,rows, size=samples_output_shape)

        dst_samples = dst_keypoints[samples_indices]
        src_samples = src_keypoints[samples_indices]

        print(f'\nIndices to pull samples from:\n {samples_indices}')
        # FOR TESTING: print(f'\nDestination Image Samples:\n {dst_samples}')
        # FOR TESTING: print(f'\nSource Image Samples:\n {src_samples}')   
        
        # min_samples randomly selected values from data
        maybe_dst_inliers = dst_samples
        maybe_src_inliers = src_samples
        also_dst_inliers = []
        also_src_inliers = [] # empty set currently, but will be any other inliers found other than the initial sample set
        this_err = 0

        # step 2: fit a model to the data such that transforming the input by the model parameters yields a close approximation to the targets
        # model parameters fitted to maybe_inliers
        maybe_model = tf_model(dst_samples, src_samples)

        # step 3: measure the error of how well ALL data fits and select the number of inliers with error less than t
        index = 0
        for point in src_keypoints:
            if point not in maybe_src_inliers:
                # represent as a homogenous point and transpose to turn into a vector form
                homogenous_point = (np.array([point[0], point[1], 1])).T
                #FOR TESTING: print(f'\nPoint vs Homogenous Source Point:\n {point} vs {homogenous_point}')
                
                # transform the current point to approximate where it would be in the destination image
                estimate_dpoint = maybe_model @ homogenous_point
                estimate_dpoint /= estimate_dpoint[2] # rehomogenize the point and divide by w
                dst_point = dst_keypoints[index]
                homogenous_dst_point = (np.array([dst_point[0], dst_point[1], 1])).T

                # compute the error between this approximation and the actual matching point
                difference = homogenous_dst_point - estimate_dpoint
                error = np.linalg.norm(difference)
                this_err += error # a measure of how well better_model fits these points

                if error < threshold_boundary:
                    also_src_inliers.append(point)
                    also_dst_inliers.append(dst_point)
        
            if len(also_src_inliers) > d:
                # this implies we may have found a good model
                # now test how good it is
                all_dst_inliers = np.append(maybe_dst_inliers, values=also_dst_inliers, axis=0)
                all_src_inliers = np.append(maybe_src_inliers, values=also_src_inliers, axis=0)

                # FOR TESTING: print(f'\nAll destination inliers found for iteration {i} with shape {all_dst_inliers.shape}:\n {all_dst_inliers}')
                # FOR TESTING: print(f'\nAll source inliers found for iteration {i} with shape {all_src_inliers.shape}:\n {all_src_inliers}')


                better_model = tf_model(all_dst_inliers, all_src_inliers) # model parameters fitted to all points in maybe_inliers and also_inliers
                
                # step 4: if the error is lower than the previous best error, fit a new model to these inliers
                if this_err < best_err:
                    best_fit = better_model
                    best_err = this_err
                    best_inliers = (all_dst_inliers, all_src_inliers)

            index += 1
        
        i += 1
            
    return best_fit, best_inliers

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
    # fig, ax = plt.figure(figsize=(8, 4))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # ax1.imshow(dst_img, cmap='gray')
    # ax2.imshow(src_img, cmap='gray')

    fig, ax = plt.subplots(1,2) #1 row, 2 columns for side-by-side subplots
    ax[0].set_title('Destination Image')
    ax[0].imshow(dst_img, cmap='gray')
    ax[1].set_title('Source Image')
    ax[1].imshow(src_img, cmap='gray')

    plt.show()

    # 1 Keypoint Detection - extract features that describe the object in the image
    # descriptor - describe/define features despite orientation, size, position, etc.
    # can do harris corner detection
    dst_detector = SIFT()
    dst_detector.detect_and_extract(dst_img) #detects keypoints and extracts descriptors
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
    matches = np.asarray(match(dst_detector, src_detector)) #store the results as a numpy array for slicing later
    print(f'\n{len(matches)} Matches found (i,j):')
    print(matches)

    # select the points in the destination image that match with the source image
    dst_matches = dst_keypoints[matches[:,0]] # select the dst_keypoints at the indices i listed in the first column of matches
    src_matches = src_keypoints[matches[:,1]] # select the src_keypoints at the indices j listed in the second column of matches

    # 2.1 Plot Keypoint Matches
    plot_matches(dst_img_rgb, src_img_rgb, dst_matches, src_matches)

    #3.3 RANSAC
    best_fit_mtx, best_inliers = ransac(dst_matches, src_matches, compute_affine_transform, iterations=100, min_samples=3, threshold_boundary=1, d=40)

    if np.any(best_fit_mtx): # a best fit model was found
        best_dst_inliers = best_inliers[0]
        best_src_inliers = best_inliers[1]

        dst_best = dst_keypoints[matches[best_dst_inliers, 0]][:, ::-1]
        src_best = src_keypoints[matches[best_src_inliers, 1]][:, ::-1]

        print(f'\nBest fit matrix found:\n {best_fit_mtx}')
        print(f'\nBest destination inliers:\n {dst_best}')
        print(f'\nBest source inliers:\n {src_best}')

        plot_matches(dst_img_rgb, src_img_rgb, dst_best, src_best)

        # Compute output shape
        # transform the corners of source image by the inverse of the best fit model
        rows, cols = dst_img.shape
        corners = np.array([
                            [0, 0, 1],
                            [cols, 0, 1],
                            [0, rows, 1],
                            [cols, rows, 1]
                        ])
        
        corners_proj = (best_fit_mtx @ corners.T).T # transform corner points
        
        #divide by w
        corners_proj[:, :2] /= corners_proj[:, 2, None] # None value ensures both x and y values are divided by w
        
        all_corners = np.vstack((corners_proj[:, :2], corners[:, :2])) # stack projected corners with original corners - to determine actual new image boundaries are that are stretched in order to accomodate
        corner_min = np.min(all_corners, axis=0)
        corner_max = np.max(all_corners, axis=0)
        output_shape = (corner_max - corner_min)
        output_shape = np.ceil(output_shape[::-1]).astype(int) # ::-1 takes last dimension and swaps x, y (since numpy treats as rows then cols)
        print('\nOutput shape:')
        print(output_shape)

        offset = SimilarityTransform(translation=-corner_min)
        dst_warped = warp(dst_img_rgb, offset.inverse, output_shape=output_shape) # still need to warp destination image because destination image is with respect to its original image size so translate into leftmost corner

        tf_img = warp(src_img_rgb, (best_fit_mtx + offset), output_shape=output_shape)

        # combine the images
        foreground_pixels = tf_img[tf_img > 0]
        dst_warped[tf_img > 0] = foreground_pixels

        # plot the result
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(dst_warped)

        plt.show()
    else:
        print(f'\nA best fit was not found')

if __name__ == "__main__":
    main()