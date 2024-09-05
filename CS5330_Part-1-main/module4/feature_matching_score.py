import numpy as np
import matplotlib.pyplot as plt
import cv2


# Add scores
def draw_matches_with_scores(img_match,img1,kp1,img2,kp2,matches,max_matches):
    
    # Adding scores to the image
    # 


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
    
    # Draw top matches
    max_matches = 5
    matching_result = cv2.drawMatches(image1, keypoints1, 
                                      image2, keypoints2, 
                                      matches[:max_matches], None, 
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matching_result = draw_matches_with_scores(matching_result, 
                                      image1, keypoints1, 
                                      image2, keypoints2, matches, max_matches)

    # Display the matching results
    plt.imshow(matching_result, cmap='gray')
    plt.axis('off')
    plt.show()
    
if __name__ == '__main__':
    main()
