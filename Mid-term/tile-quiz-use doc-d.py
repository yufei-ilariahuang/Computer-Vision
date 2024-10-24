import cv2
import numpy as np
import os
import argparse
import numpy as np


def read_images_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
    images = [cv2.imread(os.path.join(folder_path, img)) for img in image_files]
    return images


class ImageStitcher:
    def __init__(self, response_threshold=0.02, match_ratio=0.4):
        """
        Initialize the stitcher with SIFT detector and FLANN matcher
        
        Args:
            response_threshold (float): Threshold for filtering keypoints (default: 0.02)
            match_ratio (float): Ratio test threshold for matching (default: 0.4)
        """
        self.response_threshold = response_threshold
        self.match_ratio = match_ratio
        # SIFT is recommended for this case due to the text and detailed features
        self.detector = cv2.SIFT_create()
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    def determine_image_positions(self, images):
        """
        Automatically determine the position of each image in the 2x2 grid
        using feature matching to find overlapping regions
        """
        # Calculate features and matches between all pairs
        matches_info = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                print(f"\nAnalyzing pair {i+1} and {j+1}")
            
                # Convert to grayscale first
                gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(images[j], cv2.COLOR_BGR2GRAY)
                
                kp1, des1 = self.detector.detectAndCompute(gray1, None)
                kp2, des2 = self.detector.detectAndCompute(gray2, None)
                print(f"Keypoints found: img{i+1}={len(kp1)}, img{j+1}={len(kp2)}")
                if des1 is not None and des2 is not None:
                    matches = self.matcher.knnMatch(des1, des2, k=2)
                    good_matches = []
                    for m, n in matches:
                        if m.distance < self.match_ratio * n.distance:
                            good_matches.append(m)
                    print(f"Good matches found: {len(good_matches)}")
                    if len(good_matches) > 10:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                        
                        rel_x = np.mean(dst_pts[:, 0] - src_pts[:, 0])
                        rel_y = np.mean(dst_pts[:, 1] - src_pts[:, 1])
                        
                        print(f"Relative position: x={rel_x:.2f}, y={rel_y:.2f}")
                        
                        matches_info.append({
                            'img1_idx': i,
                            'img2_idx': j,
                            'matches': len(good_matches),
                            'rel_x': rel_x,
                            'rel_y': rel_y
                        })
        print(f"\nTotal valid pairs found: {len(matches_info)}")
        if not matches_info:
            raise ValueError("Not enough matches found between images")
        
        # Sort matches by number of good matches
        matches_info.sort(key=lambda x: x['matches'], reverse=True)
        
        # Initialize result dictionary
        result = {}
        used_indices = set()
        
        # Use the strongest horizontal match to determine top left and top right
        for match in matches_info:
            if abs(match['rel_x']) > abs(match['rel_y']):  # horizontal pair
                idx1, idx2 = match['img1_idx'], match['img2_idx']
                if match['rel_x'] > 0:  # idx2 is to the right of idx1
                    result['top_left'] = images[idx1]
                    result['top_right'] = images[idx2]
                else:  # idx1 is to the right of idx2
                    result['top_left'] = images[idx2]
                    result['top_right'] = images[idx1]
                used_indices.add(idx1)
                used_indices.add(idx2)
                break
        
        if 'top_left' not in result:
            raise ValueError("Could not determine top row images")
        
        # Find remaining images (bottom row)
        remaining_indices = set(range(len(images))) - used_indices
        remaining_images = [images[i] for i in remaining_indices]
        
        # Find which remaining image matches better with top_left
        max_matches = 0
        bottom_left_idx = None
        
        for i, img in enumerate(remaining_images):
            gray1 = cv2.cvtColor(result['top_left'], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            kp1, des1 = self.detector.detectAndCompute(gray1, None)
            kp2, des2 = self.detector.detectAndCompute(gray2, None)
            
            if des1 is not None and des2 is not None:
                matches = self.matcher.knnMatch(des1, des2, k=2)
                good_matches = [m for m, n in matches if m.distance < self.match_ratio * n.distance]
                
                if len(good_matches) > max_matches:
                    max_matches = len(good_matches)
                    bottom_left_idx = i
        
        if bottom_left_idx is not None:
            result['bottom_left'] = remaining_images[bottom_left_idx]
            result['bottom_right'] = remaining_images[1 - bottom_left_idx]  # The other remaining image
        else:
            # If matching fails, just assign arbitrarily
            result['bottom_left'] = remaining_images[0]
            result['bottom_right'] = remaining_images[1]
        
        return result
    
    def stitch_images(self, images):
        """
        Stitch images based on the number of input images
        """
        num_images = len(images)
        
        if num_images == 4:
            return self.stitch_four_images(images)
        elif num_images == 9:
            return self.stitch_nine_images(images)
        else:
            raise ValueError(f"Unsupported number of images: {num_images}")

    def stitch_four_images(self, images):
        """
        Stitch four images arranged in a 2x2 grid
        
        Args:
            images: List of 4 images (will be automatically arranged)
        """
        if len(images) != 4:
            raise ValueError("Exactly 4 images required")

        # Determine positions
        arranged_images = self.determine_image_positions(images)
        
        # Stitch in the correct order
        top_row = self.stitch_pair(arranged_images['top_left'], arranged_images['top_right'])
        bottom_row = self.stitch_pair(arranged_images['bottom_left'], arranged_images['bottom_right'])
        final_image = self.stitch_vertical(top_row, bottom_row)
        
        return self.post_process(final_image)
    
    def stitch_pair(self, img1, img2):
        print("\nStarting pair stitching...")
        print(f"Input shapes: img1={img1.shape}, img2={img2.shape}")
        
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        print(f"Keypoints found: img1={len(kp1)}, img2={len(kp2)}")
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_ratio * n.distance:
                good_matches.append(m)
        
        print(f"Good matches found: {len(good_matches)}")
        if len(good_matches) < 4:
            print("Not enough good matches found. Returning the first image.")
            return img1
        try:
            # Get matching points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            print("Homography matrix found:", H)
            
            # Get dimensions
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            print(f"Original dimensions: img1=({w1}x{h1}), img2=({w2}x{h2})")
            
            # Create points for img2 corners
            corners = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)
            
            # Transform corners through homography
            transformed_corners = cv2.perspectiveTransform(corners, H)
            print("Transformed corners:", transformed_corners.reshape(-1, 2))
            
            # Get corners of both images
            all_corners = np.vstack((
                transformed_corners,
                np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
            ))
            
            # Find min and max coordinates
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
            print(f"Coordinate ranges: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]")
            
            # Calculate output size
            output_width = xmax - xmin + 1
            output_height = ymax - ymin + 1
            print(f"Output dimensions: {output_width}x{output_height}")
            
            # Create translation matrix
            translation = np.array([
                [1, 0, -xmin],
                [0, 1, -ymin],
                [0, 0, 1]
            ])
            
            # Combine transformation matrices
            final_matrix = translation.dot(H)
            
            # Create output canvas
            result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # Warp second image
            warped_img2 = cv2.warpPerspective(img2, final_matrix, (output_width, output_height))
            
            # Calculate coordinates for first image
            x_offset = -xmin
            y_offset = -ymin
            
            # Create a mask for better blending
            mask = np.zeros((output_height, output_width), dtype=np.uint8)
            
            # Define the region for img1
            img1_region = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            y_start = max(0, y_offset)
            y_end = min(output_height, y_offset + h1)
            x_start = max(0, x_offset)
            x_end = min(output_width, x_offset + w1)
            
            print(f"Placing img1 at y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")
            
            # Place img1 in its region
            img1_region[y_start:y_end, x_start:x_end] = img1[
                max(0, -y_offset):min(h1, output_height-y_offset),
                max(0, -x_offset):min(w1, output_width-x_offset)
            ]
            
            # Create a mask for blending
            mask[y_start:y_end, x_start:x_end] = 255
            
            # Blend the images
            mask_3d = np.stack([mask/255.0]*3, axis=2)
            result = img1_region * mask_3d + warped_img2 * (1 - mask_3d)
            
            print(f"Final output shape: {result.shape}")
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in stitch_pair: {str(e)}")
        print("Returning the first image as a fallback.")
        return img1
        
    def post_process(self, image):
        """Post-process the stitched image"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find non-black regions
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main content)
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Crop to content
            cropped = image[y:y+h, x:x+w]
            
            # Optional: Enhance image
            cropped = cv2.convertScaleAbs(cropped, alpha=1.1, beta=0)
            
            return cropped
        
        return image
    
    def stitch_nine_images(self, images):
        """
        Stitch nine images arranged in a 3x3 grid
        
        Args:
            images: List of 9 images in order [
                top_left,    top_middle,    top_right,
                middle_left, middle_middle, middle_right,
                bottom_left, bottom_middle, bottom_right
            ]
        
        Returns:
            Stitched image
        """
        if len(images) != 9:
            raise ValueError(f"Expected 9 images, got {len(images)}")
        
        try:
            # Step 1: Stitch each row horizontally
            print("\nStitching top row...")
            top_row = self.stitch_three_horizontal(
                images[0],  # top_left
                images[1],  # top_middle
                images[2]   # top_right
            )
            print("Top row complete. Shape:", top_row.shape)
            
            print("\nStitching middle row...")
            middle_row = self.stitch_three_horizontal(
                images[3],  # middle_left
                images[4],  # middle_middle
                images[5]   # middle_right
            )
            print("Middle row complete. Shape:", middle_row.shape)
            
            print("\nStitching bottom row...")
            bottom_row = self.stitch_three_horizontal(
                images[6],  # bottom_left
                images[7],  # bottom_middle
                images[8]   # bottom_right
            )
            print("Bottom row complete. Shape:", bottom_row.shape)
            
            # Step 2: Stitch the rows vertically
            print("\nStitching rows vertically...")
            print("Stitching top and middle rows...")
            top_half = self.stitch_vertical(top_row, middle_row)
            print("Top half complete. Shape:", top_half.shape)
            
            print("Stitching with bottom row...")
            final_image = self.stitch_vertical(top_half, bottom_row)
            print("Vertical stitching complete. Shape:", final_image.shape)
            
            # Step 3: Post-process the result
            print("\nPost-processing final image...")
            result = self.post_process(final_image)
            print("Post-processing complete. Final shape:", result.shape)
            
            return result
            
        except Exception as e:
            print(f"Error in nine image stitching: {str(e)}")
            print("Returning a simple grid of the original images.")
            # Create a 3x3 grid of the original images
            rows = [np.hstack(images[i:i+3]) for i in range(0, 9, 3)]
            return np.vstack(rows)

    def stitch_three_horizontal(self, img_left, img_middle, img_right):
        """
        Stitch three images horizontally from left to right
        
        Args:
            img_left: Left image
            img_middle: Middle image
            img_right: Right image
        
        Returns:
            Stitched image
        """
        try:
            # First stitch left and middle images
            print("Stitching left and middle images...")
            left_pair = self.stitch_pair(img_left, img_middle)
            print("Left pair complete. Shape:", left_pair.shape)
            
            # Then stitch with the right image
            print("Stitching with right image...")
            result = self.stitch_pair(left_pair, img_right)
            print("Horizontal triple stitch complete. Shape:", result.shape)
            
            return result
            
        except Exception as e:
            print(f"Error in three image horizontal stitching: {str(e)}")
            raise

    def stitch_vertical(self, img_top, img_bottom):
        """
        Stitch two images vertically
        
        Args:
            img_top: Top image
            img_bottom: Bottom image
        
        Returns:
            Stitched image
        """
        try:
            print(f"Vertical stitching images of shapes {img_top.shape} and {img_bottom.shape}")
            
            # First, ensure images are the same width
            if img_top.shape[1] != img_bottom.shape[1]:
                print("Adjusting image widths...")
                max_width = max(img_top.shape[1], img_bottom.shape[1])
            
                # Pad images to match the maximum width
                if img_top.shape[1] < max_width:
                    img_top = cv2.copyMakeBorder(img_top, 0, 0, 0, max_width - img_top.shape[1], cv2.BORDER_CONSTANT, value=0)
                if img_bottom.shape[1] < max_width:
                    img_bottom = cv2.copyMakeBorder(img_bottom, 0, 0, 0, max_width - img_bottom.shape[1], cv2.BORDER_CONSTANT, value=0)
                print("Adjusted shapes:", img_top.shape, img_bottom.shape)
                # Convert to grayscale for feature matching
                gray1 = cv2.cvtColor(img_top, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img_bottom, cv2.COLOR_BGR2GRAY)
                
                # Detect keypoints and compute descriptors
                kp1, des1 = self.detector.detectAndCompute(gray1, None)
                kp2, des2 = self.detector.detectAndCompute(gray2, None)
                print(f"Found {len(kp1)} and {len(kp2)} keypoints")
                
                # Match features
                matches = self.matcher.knnMatch(des1, des2, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < self.match_ratio * n.distance:
                        good_matches.append(m)
                print(f"Found {len(good_matches)} good matches")
                
                if len(good_matches) < 4:
                    print("Not enough good matches. Stacking images vertically.")
                    return np.vstack((img_top, img_bottom))
                
                # Get matching points
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                
                # Get dimensions
                h1, w1 = img_top.shape[:2]
                h2, w2 = img_bottom.shape[:2]
                
                # Transform corners
                corners = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, H)
                
                # Get all corners
                all_corners = np.vstack((
                    transformed_corners,
                    np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
                ))
                
                # Calculate bounds
                [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
                [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
                
                # Create translation matrix
                translation = np.array([
                    [1, 0, -xmin],
                    [0, 1, -ymin],
                    [0, 0, 1]
                ])
                
                # Combine transformations
                final_matrix = translation.dot(H)
                
                # Calculate output size
                width = xmax - xmin + 1
                height = ymax - ymin + 1
                
                # Create output canvas
                result = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Warp bottom image
                warped_bottom = cv2.warpPerspective(img_bottom, final_matrix, (width, height))
                
                # Place top image
                x_offset = -xmin
                y_offset = -ymin
                
                # Create mask for blending
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # Define region for top image
                top_region = np.zeros((height, width, 3), dtype=np.uint8)
                y_start = max(0, y_offset)
                y_end = min(height, y_offset + h1)
                x_start = max(0, x_offset)
                x_end = min(width, x_offset + w1)
                
                # Place top image
                top_region[y_start:y_end, x_start:x_end] = img_top[
                    max(0, -y_offset):min(h1, height-y_offset),
                    max(0, -x_offset):min(w1, width-x_offset)
                ]
                
                # Create blending mask
                mask[y_start:y_end, x_start:x_end] = 255
                mask_3d = np.stack([mask/255.0]*3, axis=2)
                
                # Blend images
                result = top_region * mask_3d + warped_bottom * (1 - mask_3d)
                
                return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in vertical stitching: {str(e)}")
            print("Stacking images vertically as a fallback.")
            return np.vstack((img_top, img_bottom))
    
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stitch 9 images and display.')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Path to the folder containing images')
    args = parser.parse_args()

    # Read images from the given folder
    images = read_images_from_folder(args.directory)

    # Create stitcher instance (using SIFT for better accuracy with text)
    stitcher = ImageStitcher()
    
    # Stitch images
    try:
        # Check image arrangement
        arranged_images = stitcher.determine_image_positions(images)
        print("\nArranged images positions:")
        for pos, img in arranged_images.items():
            print(f"{pos}: shape {img.shape}")

        # Check stitching steps
        print("\nStarting stitching process...")
        top_row = stitcher.stitch_pair(arranged_images['top_left'], arranged_images['top_right'])
        print("Top row stitching completed. Shape:", top_row.shape)
        
        bottom_row = stitcher.stitch_pair(arranged_images['bottom_left'], arranged_images['bottom_right'])
        print("Bottom row stitching completed. Shape:", bottom_row.shape)
        
        final_image = stitcher.stitch_vertical(top_row, bottom_row)
        print("Vertical stitching completed. Shape:", final_image.shape)
        
        result = stitcher.post_process(final_image)
        print("Post-processing completed. Final shape:", result.shape)
        
        # Save and display result
        cv2.imwrite("{}.jpg".format(args.directory), result)
        print("\nResult saved successfully")
        
        cv2.imshow("Stitched Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during stitching: {str(e)}")

if __name__ == "__main__":
    main()