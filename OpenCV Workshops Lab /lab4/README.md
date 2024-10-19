
# Lab 4: Feature Detection and Matching for Dog Images

## Introduction
This lab focuses on applying feature detection and matching techniques to dog images using OpenCV. We explore Harris Corner Detection, SIFT, and FLANN matching algorithms to identify and match distinctive features in Doberman images.

## Feature Detection

### Harris Corner Detection
Implement Harris Corner Detection with customizable parameters:
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
![Harris Corner Detection](https://github.com/user-attachments/assets/1316e7ad-e05f-4e6c-b8d5-2fdae7061d11)
- SIFT keypoints drawn on separate images
![Image2 keypoints](https://github.com/user-attachments/assets/0f8b0e1f-8900-4263-a548-0a460819d9c9)
![Image1 keypoints](https://github.com/user-attachments/assets/d05d9198-b105-4141-83ef-22cc1d941dc8)
- Matches displayed with green lines connecting corresponding features
![Matches](https://github.com/user-attachments/assets/6e759f5a-3287-4137-84ca-6f13a86dea7c)

## Parameters and Tuning
- Adjustable parameters for Harris corner detection
- Configurable SIFT parameters for feature detection
- Tunable FLANN matching parameters
- Adjustable ratio for match filtering

