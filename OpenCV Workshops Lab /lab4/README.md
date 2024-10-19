
# Lab 4: Feature Detection and Matching for Dog Images

## Introduction
This lab focuses on applying feature detection and matching techniques to dog images using OpenCV. We explore Harris Corner Detection, SIFT, and FLANN matching algorithms to identify and match distinctive features in Doberman images.

## Feature Detection

### Harris Corner Detection
We implement Harris Corner Detection with customizable parameters:
- `block_size`: Size of the neighborhood for corner detection
- `ksize`: Aperture parameter for the Sobel operator
- `k`: Harris detector free parameter
- `threshold_ratio`: Determines the strength threshold for corner consideration
- `max_corners`: Limits the number of detected corners

### SIFT (Scale-Invariant Feature Transform)
SIFT is used for detecting and describing local features in the dog images:
- `nfeatures`: Number of best features to retain
- `nOctaveLayers`: Number of layers in each octave
- `contrastThreshold`: Filter out weak features
- `edgeThreshold`: Filter out edge-like features
- `sigma`: The sigma of the Gaussian applied to the input image

## Feature Matching

### FLANN (Fast Library for Approximate Nearest Neighbors)
FLANN is implemented for efficient matching of SIFT descriptors:
- Uses KD-tree algorithm for indexing
- Configurable number of trees and checks for balancing speed and accuracy

## Image Preprocessing
- Image resizing while maintaining aspect ratio
- Gaussian blur application for noise reduction
- Edge enhancement using Laplacian

## Code Implementation
Key components of our implementation:
- `harris_corner_detection`: Detects and visualizes corners
- `sift_detection`: Extracts SIFT features and descriptors
- `flann_matching`: Performs feature matching
- `filter_matches`: Filters matches based on a ratio test
- `resize_image`: Resizes images for consistent processing

## Visualization
- Harris corners visualized with red circles
- SIFT keypoints drawn on separate images
- Matches displayed with green lines connecting corresponding features

## Parameters and Tuning
- Adjustable parameters for Harris corner detection
- Configurable SIFT parameters for feature detection
- Tunable FLANN matching parameters
- Adjustable ratio for match filtering

