Hoang Ho - 1001654608
CSE 4310-001 Fundamentals of Computer Vision
Assignment 2 - SIFT and RANSAC
Due March 11, 2022 by 11:59 PM

references: https://github.com/ajdillhoff/CSE4310/blob/main/ransac.ipynb
            https://dillhoffaj.utasites.cloud/posts/scale_invariant_feature_transforms/
            https://dillhoffaj.utasites.cloud/posts/random_sample_consensus/
            https://dillhoffaj.utasites.cloud/posts/image_features/
            https://en.wikipedia.org/wiki/Random_sample_consensus

ensure latest version of scikit-image with SIFT installed:
	conda  search scikit-image -c conda-forge
	conda install scikit-image=0.19.2 -c conda-forge

About inputs:
	Since I did not code asking for user input, you can find the lines to open two images in the main function at lines 338 and 339 and edit them to open different image samples as needed. You can also find the line that calls the ransac function and adjust its parameters as needed on line 397 which is also in the main function. Some lines of code were used for testing and printing outputs, but were commented out in the end so it does not clutter up the terminal too much.

For the report for 3.4 Testing section, it is named testing-report.pdf