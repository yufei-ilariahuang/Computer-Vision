import cv2
import numpy as np
'''
Harris Corner Detector
- block_size=2: This is the size of the neighborhood considered for corner detection. A smaller value (like 2) can detect finer corners, which might be useful for capturing details in the dog's features.
- ksize=3: This is the aperture parameter for the Sobel operator. A value of 3 is often a good balance between detecting edges and noise sensitivity.
- k=0.04: This is the Harris detector free parameter. The default of 0.04 often works well, but you can adjust it slightly (e.g., 0.03-0.06) to fine-tune sensitivity.
- threshold_ratio=0.01: This determines how strong a corner needs to be to be considered. Lowering this value (e.g., 0.005) will detect more corners, while increasing it (e.g., 0.02) will be more selective.
- max_corners=500: This limits the number of corners detected to prevent over-cluttering the image. Adjust based on how many corners you want to visualize.
'''
def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold_ratio=0.001, max_corners=500):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Enhance edges
    gray = cv2.Laplacian(gray, cv2.CV_64F)
    gray = cv2.convertScaleAbs(gray)
    '''
    cv2.cornerHarris(src, dest, blockSize, kSize, freeParameter, borderType)
        • Src – input image
        • Dest – image to store the Harris detector responses
        • blockSize – neighborhood size
        • Ksize – Aperture parameter for Sobel() operator
        • freeParameter – Harris detector free parameter
        • borderType – Pixel extrapolation method
    '''
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    '''
    cv2.dilate()
        • Dilates an image by using a specific structuring element that determines the shape of a pixel neighborhood which the maximum is taken:
        https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c

    '''
    dst = cv2.dilate(dst, None)
    # Threshold for corner response
    threshold = threshold_ratio * dst.max()
    corner_mask = dst > threshold
    
    # Find coordinates of corners
    corners = np.argwhere(corner_mask)
    
    # Sort corners by strength (descending order)
    corner_strengths = dst[corner_mask]
    sorted_indices = np.argsort(corner_strengths)[::-1]
    corners = corners[sorted_indices]
    
    # Limit the number of corners
    corners = corners[:max_corners]
    
    # Draw circles at corner positions
    result = image.copy()
    for y, x in corners:
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
    
    return result

'''
SIFT (Scale-Invariant Feature Transform)
• CV algorithm to detect, describe, and match local features in images
• An object is recognized in a new image by individually comparing each
feature form the new image to its database and finding candidate
matching features based on Euclidean distance of their feature vectors
'''
def sift_detection(image, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, 
                           contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors
'''
Feature Matching with FLANN
• FLANN (Fast Library for Approximate Nearest Neighbors) is an
algorithm that is used to match feature descriptors between
images
• Contains a collection of algorithms to work best for nearest
neighbor search and a system for automatically choosing the best
algorithm and optimum parameters based on the dataset
• FLANN is written in C++ and contains bindings for C, MATLAB,
Python, and Ruby
'''
def flann_matching(descriptors1, descriptors2):
    '''
    FLANN_INDEX_KDTREE: Algorithm for indexing, using KD-tree.
        • parameter specifies that the KD-tree algorithm should be used for indexing. KD-trees are a data structure that efficiently organizes points
in a space, which is useful for nearest neighbor searches.
        • Parameters:
            • algorithm: Specifies the algorithm to use for indexing. In this case, FLANN_INDEX_KDTREE is used.
            • trees: Specifies the number of trees to be used in the KD-tree. A higher number of trees may result in more accurate but slower searches
    '''
    FLANN_INDEX_KDTREE = 1
    '''
    index_params: Specifies the algorithm and number of trees.
        • The index_params dictionary contains parameters that define how the feature descriptors will be indexed.
        • Parameters:
            • algorithm: Specifies the algorithm to use for indexing. In this case, FLANN_INDEX_KDTREE is used.
            •  trees: Specifies the number of trees to be used in the KD-tree. A higher number of trees may result in more accurate but slower searches
    '''
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    '''
    search_params: Controls the number of checks during the search.
        • The search_params dictionary contains parameters that control the search behavior during the nearest neighbor search.
        • Parameters:
            • checks: Specifies the number of times the tree(s) will be traversed recursively. A higher value results in a more exhaustive search, which may increase
        accuracy at the cost of speed. The default is often around 50.

    '''
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    '''
    FLANN Matching: flann.knnMatch()
        • The knnMatch() function performs K-Nearest Neighbor matching between the descriptors of the two images. It returns the k best matches for each descriptor.
        • Parameters:
            • descriptors1: The descriptors from the first image.
            • descriptors2: The descriptors from the second image.
            • k: The number of nearest neighbors to find for each descript
    '''
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    return matches

def filter_matches(matches, ratio=0.75):
    good_matches = []
    for match in matches:
        if len(match) >= 2:
            m, n = match[:2]
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    return good_matches
def resize_image(image, target_size):
    """
    Resize the image to a target size while maintaining aspect ratio.
    
    :param image: Input image
    :param target_size: Target size for the longer side of the image
    :return: Resized image
    """
    h, w = image.shape[:2]
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)




