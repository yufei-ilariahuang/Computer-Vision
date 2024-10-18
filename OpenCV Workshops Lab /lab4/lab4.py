import cv2
from feature_detection import harris_corner_detection, sift_detection, flann_matching, filter_matches

def lab4():
    # Load images
    img1 = cv2.imread('../image/scene.jpg')
    img2 = cv2.imread('../image/object.jpg')

    # Harris Corner Detection
    harris_img = harris_corner_detection(img1.copy())
    cv2.imshow('Harris Corner Detection', harris_img)

    # SIFT Detection
    sift_img1, keypoints1, descriptors1 = sift_detection(img1)
    sift_img2, keypoints2, descriptors2 = sift_detection(img2)
    cv2.imshow('SIFT Detection - Scene', sift_img1)
    cv2.imshow('SIFT Detection - Object', sift_img2)

    # FLANN Matching
    matches = flann_matching(descriptors1, descriptors2)
    good_matches = filter_matches(matches)

    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('FLANN Matching', img_matches)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab4()