
import cv2
import numpy as np
from .feature_detection import harris_corner_detection, sift_detection, flann_matching, filter_matches, resize_image

def draw_matches_centered(img1, kp1, img2, kp2, matches, color=None):
    # Create a new output image that concatenates the two images
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')
    out[:rows1, :cols1, :] = img1
    out[:rows2, cols1:cols1+cols2, :] = img2

    # For each pair of points we have between both images
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        cv2.line(out, (int(x1), int(y1)), (int(x2)+cols1, int(y2)), color, 1)

    return out

def lab4():
    # Load images
    img1 = cv2.imread('image/w7.jpg')
    img2 = cv2.imread('image/w6.jpg')

    # Resize images
    target_size = 800  # You can adjust this value
    img1_resized = resize_image(img1, target_size)
    img2_resized = resize_image(img2, target_size)

    # Harris Corner Detection
    harris_img = harris_corner_detection(img1_resized.copy())
    cv2.imshow('Harris Corner Detection', harris_img)

    # SIFT Detection with adjusted parameters
    keypoints1, descriptors1 = sift_detection(img1_resized, nfeatures=0, contrastThreshold=0.01)
    keypoints2, descriptors2 = sift_detection(img2_resized, nfeatures=0, contrastThreshold=0.01)

    # FLANN Matching
    matches = flann_matching(descriptors1, descriptors2)
    good_matches = filter_matches(matches, ratio=0.75)

    # Draw matches with centered lines
    img_matches = draw_matches_centered(img1_resized, keypoints1, img2_resized, keypoints2, good_matches, color=(0, 255, 0))

    # Draw keypoints on separate images
    img1_keypoints = cv2.drawKeypoints(img1_resized, keypoints1, None, color=(0, 0, 255),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_keypoints = cv2.drawKeypoints(img2_resized, keypoints2, None, color=(0, 0, 255),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display results
    cv2.imshow('Image 1 Keypoints', img1_keypoints)
    cv2.imshow('Image 2 Keypoints', img2_keypoints)
    cv2.imshow('Matches', img_matches)

    print(f"Total keypoints in Image 1: {len(keypoints1)}")
    print(f"Total keypoints in Image 2: {len(keypoints2)}")
    print(f"Number of good matches: {len(good_matches)}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab4()