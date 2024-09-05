import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_cv_img(template_image, output_image, matched_image):
    """     
    Converts an image from BGR to RGB and plots     
    """

    fig, ax = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [1, 4, 5]})

    ax[0].imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Template')
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Output Image')
    ax[1].axis('off')

    ax[2].imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Matched Image')
    ax[2].axis('off')

    plt.show()

def main():
    # Load the main image and the template image
    image = cv2.imread('../images/livingroom1.png')
    template = cv2.imread('../images/chair.png')

    # Convert the main image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT keypoints and descriptors in the images
    keypoints1, descriptors1 = sift.detectAndCompute(gray_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_template, None)

    # Use BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Match descriptors and sort them in the order of their distance
    matches = bf.knnMatch(descriptors2, descriptors1, k=2)

    # Apply ratio test as per Lowe's paper
    good_matches = []
    threshold=0.75
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    # Draw matches
    matched_image = cv2.drawMatches(gray_template, keypoints2, 
                                    image, keypoints1, 
                                    good_matches, None, 
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # If enough matches are found, we extract the location of matched keypoints in both images
    if len(good_matches) > 10:
        src_pts = np.float32([keypoints2[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Compute homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get the dimensions of the template image
        h, w = gray_template.shape

        # Get the coordinates of the corners of the template image
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        # Transform the corners of the template image to the main image
        dst = cv2.perspectiveTransform(pts, H)

        # Draw the transformed corners on the main image
        image = cv2.polylines(image, [np.int32(dst)], True, 
                                   (0, 255, 0), 3, cv2.LINE_AA)

    # Do plot
    plot_cv_img(template, image, matched_image)

if __name__ == '__main__':
    main()
