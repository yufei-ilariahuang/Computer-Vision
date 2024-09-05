import cv2 
import matplotlib.pyplot as plt
import numpy as np

def stitch_images_sift(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect SIFT keypoints and descriptors
    #

    # Match features using the BFMatcher with the L2 norm
    #

    # Sort matches by distance (best matches first)
    #

    # Extract the matched keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Compute the homography matrix using RANSAC
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Warp the second image to align with the first image
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    warped_image2 = cv2.warpPerspective(image2, H, (width1 + width2, height1))

    # Combine the images to create a panorama
    panorama = np.zeros((height1, width1 + width2, 3), dtype=np.uint8)
    panorama[0:height1, 0:width1] = image1
    panorama[0:height1, 0:width1 + width2] = np.maximum(panorama[0:height1, 0:width1 + width2], warped_image2)

    return panorama

def main():
    # Load images
    img1 = cv2.imread('../images/mountain1.png')
    img2 = cv2.imread('../images/mountain2.png')

    # Stitch the images
    stitched_image = stitch_images_sift(img1, img2)

    # display the stitched image
    plt.imshow(stitched_image, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
