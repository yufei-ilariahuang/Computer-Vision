import cv2
import os
import argparse
import numpy as np
def find_keypoints_and_descriptors(images):
    imgs_keypoints = []
    imgs_descriptors = []
    for i, img in enumerate(images):
        sift = cv2.SIFT_create()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray_img, None)
        response_threshold = 0.02
        filtered_keypoints = [kp for kp in keypoints if kp.response>response_threshold]
        image_with_keypoints = cv2.drawKeypoints(img, filtered_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow(f'Image with Keypoints {i}', image_with_keypoints)
        imgs_keypoints.append(keypoints)
        imgs_descriptors.append(descriptors)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return imgs_keypoints, imgs_descriptors

def match_keypoints(img_1, keypoints_1, descriptor_1, img_2, keypoints_2, descriptor_2, index_1, index_2):
    FLANN_INDEX_KDTREE =1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE,tree=5)
    search_params = dict(checks=50)
    flann=cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptor_1, descriptor_2, k=2)
    ratio=0.4
    good_matches = []
    for m,n in matches:
        if m.distance < ratio* n.distance:
            good_matches.append(m)
    visual_matches = cv2.drawMatches(img_1,keypoints_1,img_2, keypoints_2, good_matches, None, cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(f'Matches {index_1}_{index_2}', visual_matches)
    cv2.waitKey(0)
    return good_matches


def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    images = [cv2.imread(os.path.join(folder_path, img)) for img in image_files]
    return images

def stitch_images(images):
    """
    Stitch four images arranged in a 2x2 grid using SIFT features and homography.
    Args:
        images: List of 4 images [top_left, top_right, bottom_left, bottom_right]
    Returns:
        Stitched image
    """
    if len(images) != 4:
        print("Need exactly 4 images")
        return None
        
    # Get keypoints and descriptors for all images
    imgs_keypoints, imgs_descriptors = find_keypoints_and_descriptors(images)
    
    # First stitch horizontal pairs
    # Match and stitch top row (images 0 and 1)
    matches_top = match_keypoints(
        images[0], imgs_keypoints[0], imgs_descriptors[0],
        images[1], imgs_keypoints[1], imgs_descriptors[1],
        0, 1
    )
    
    if len(matches_top) < 4:
        print("Not enough matches in top row")
        return None
        
    # Get matching points for top row
    src_pts_top = np.float32([imgs_keypoints[0][m.queryIdx].pt for m in matches_top]).reshape(-1, 1, 2)
    dst_pts_top = np.float32([imgs_keypoints[1][m.trainIdx].pt for m in matches_top]).reshape(-1, 1, 2)
    
    # Calculate homography for top row
    H_top, _ = cv2.findHomography(dst_pts_top, src_pts_top, cv2.RANSAC, 5.0)
    
    # Stitch top row
    h1, w1 = images[0].shape[:2]
    h2, w2 = images[1].shape[:2]
    top_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    top_transformed = cv2.perspectiveTransform(top_corners, H_top)
    [xmin_top, ymin_top] = np.int32(top_transformed.min(axis=0).ravel() - 0.5)
    [xmax_top, ymax_top] = np.int32(top_transformed.max(axis=0).ravel() + 0.5)
    
    translation_top = np.array([[1, 0, -xmin_top], [0, 1, -ymin_top], [0, 0, 1]])
    H_top = translation_top.dot(H_top)
    
    top_row = cv2.warpPerspective(images[1], H_top, (xmax_top-xmin_top, ymax_top-ymin_top))
    top_row[-ymin_top:h1-ymin_top, -xmin_top:w1-xmin_top] = images[0]
    
    # Match and stitch bottom row (images 2 and 3)
    matches_bottom = match_keypoints(
        images[2], imgs_keypoints[2], imgs_descriptors[2],
        images[3], imgs_keypoints[3], imgs_descriptors[3],
        2, 3
    )
    
    if len(matches_bottom) < 4:
        print("Not enough matches in bottom row")
        return None
        
    # Get matching points for bottom row
    src_pts_bottom = np.float32([imgs_keypoints[2][m.queryIdx].pt for m in matches_bottom]).reshape(-1, 1, 2)
    dst_pts_bottom = np.float32([imgs_keypoints[3][m.trainIdx].pt for m in matches_bottom]).reshape(-1, 1, 2)
    
    # Calculate homography for bottom row
    H_bottom, _ = cv2.findHomography(dst_pts_bottom, src_pts_bottom, cv2.RANSAC, 5.0)
    
    # Stitch bottom row
    h3, w3 = images[2].shape[:2]
    h4, w4 = images[3].shape[:2]
    bottom_corners = np.float32([[0, 0], [0, h4], [w4, h4], [w4, 0]]).reshape(-1, 1, 2)
    bottom_transformed = cv2.perspectiveTransform(bottom_corners, H_bottom)
    [xmin_bottom, ymin_bottom] = np.int32(bottom_transformed.min(axis=0).ravel() - 0.5)
    [xmax_bottom, ymax_bottom] = np.int32(bottom_transformed.max(axis=0).ravel() + 0.5)
    
    translation_bottom = np.array([[1, 0, -xmin_bottom], [0, 1, -ymin_bottom], [0, 0, 1]])
    H_bottom = translation_bottom.dot(H_bottom)
    
    bottom_row = cv2.warpPerspective(images[3], H_bottom, (xmax_bottom-xmin_bottom, ymax_bottom-ymin_bottom))
    bottom_row[-ymin_bottom:h3-ymin_bottom, -xmin_bottom:w3-xmin_bottom] = images[2]
    
    # Now stitch top and bottom rows vertically
    # Get keypoints and descriptors for the stitched rows
    row_keypoints, row_descriptors = find_keypoints_and_descriptors([top_row, bottom_row])
    
    matches_vertical = match_keypoints(
        top_row, row_keypoints[0], row_descriptors[0],
        bottom_row, row_keypoints[1], row_descriptors[1],
        4, 5
    )
    
    if len(matches_vertical) < 4:
        print("Not enough matches between rows")
        return None
        
    # Get matching points for vertical stitching
    src_pts_vert = np.float32([row_keypoints[0][m.queryIdx].pt for m in matches_vertical]).reshape(-1, 1, 2)
    dst_pts_vert = np.float32([row_keypoints[1][m.trainIdx].pt for m in matches_vertical]).reshape(-1, 1, 2)
    
    # Calculate homography for vertical stitching
    H_vert, _ = cv2.findHomography(dst_pts_vert, src_pts_vert, cv2.RANSAC, 5.0)
    
    # Stitch rows vertically
    h_top, w_top = top_row.shape[:2]
    h_bottom, w_bottom = bottom_row.shape[:2]
    vert_corners = np.float32([[0, 0], [0, h_bottom], [w_bottom, h_bottom], [w_bottom, 0]]).reshape(-1, 1, 2)
    vert_transformed = cv2.perspectiveTransform(vert_corners, H_vert)
    [xmin_vert, ymin_vert] = np.int32(vert_transformed.min(axis=0).ravel() - 0.5)
    [xmax_vert, ymax_vert] = np.int32(vert_transformed.max(axis=0).ravel() + 0.5)
    
    translation_vert = np.array([[1, 0, -xmin_vert], [0, 1, -ymin_vert], [0, 0, 1]])
    H_vert = translation_vert.dot(H_vert)
    
    final_result = cv2.warpPerspective(bottom_row, H_vert, (xmax_vert-xmin_vert, ymax_vert-ymin_vert))
    final_result[-ymin_vert:h_top-ymin_vert, -xmin_vert:w_top-xmin_vert] = top_row
    
    # Crop black borders
    gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        final_result = final_result[y:y+h, x:x+w]
    
    return final_result

def combine_images_into_grid(images, rows=2, cols=2):
    if not images:
        print("No images found in the directory.")
        return None

    # Resize images to a standard size for uniform display
    max_height, max_width = 150, 150
    resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]

    # Create blank images to complete the grid if necessary
    blank_image = np.zeros_like(resized_images[0])
    while len(resized_images) < rows * cols:
        resized_images.append(blank_image)

    # Arrange images into rows and columns
    rows_of_images = [np.hstack(resized_images[i * cols:(i + 1) * cols]) for i in range(rows)]
    combined_image = np.vstack(rows_of_images)

    return combined_image

def display_stitched_image(image):
    if image is None:
        return

    cv2.imshow('Stitched Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stitch 9 images and display.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Path to the folder containing images')
    args = parser.parse_args()

    # Read images from the given folder
    images = read_images_from_folder(args.directory)

    # Once you start, ignore combine_images_into_grid()
    # Instead, complete stitch_image(), if needed you can add more arguments
    # at stitch_image()
    #stitched_image = combine_images_into_grid(images, rows=2, cols=2)
    stitched_image = stitch_images(images) 

    # Display the stitched image
    display_stitched_image(stitched_image)
