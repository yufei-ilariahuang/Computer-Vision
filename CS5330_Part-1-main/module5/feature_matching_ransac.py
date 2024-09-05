import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    # Load images
    image1 = cv2.imread('../images/trevi1.png')
    image2 = cv2.imread('../images/trevi2.png')
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Convert keypoints to points
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Use RANSAC to find the homography matrix
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # Select inlier matches based on RANSAC mask
    matches_mask = mask.ravel().tolist()

    # Draw top matches
    max_matches = 50 
    draw_params = dict(matchColor=(0, 255, 0), # Draw matches in green color
                       singlePointColor=None,
                       matchesMask=matches_mask, # Draw only inliers
                       flags=2)

    matching_result = cv2.drawMatches(image1, keypoints1, 
                                      image2, keypoints2, 
                                      matches, None, 
                                      **draw_params)

    # Display the matching results
    plt.imshow(matching_result, cmap='gray')
    plt.axis('off')
    plt.show()
    
if __name__ == '__main__':
    main()
