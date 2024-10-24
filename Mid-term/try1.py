import cv2
import numpy as np
import os

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
    
    def stitch_vertical(self, img1, img2):
        """
        Stitch two rows vertically with size handling
        """
        print("\nStarting vertical stitching...")
        print(f"Input shapes: top_row={img1.shape}, bottom_row={img2.shape}")
        
        # First, make the rows the same width
        max_width = max(img1.shape[1], img2.shape[1])
        
        # Pad images to match the maximum width
        if img1.shape[1] < max_width:
            diff = max_width - img1.shape[1]
            img1 = cv2.copyMakeBorder(img1, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=0)
        if img2.shape[1] < max_width:
            diff = max_width - img2.shape[1]
            img2 = cv2.copyMakeBorder(img2, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=0)
        
        print(f"After width adjustment: top_row={img1.shape}, bottom_row={img2.shape}")
        
        # Convert to grayscale for feature matching
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        print(f"Keypoints found: top={len(kp1)}, bottom={len(kp2)}")
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.match_ratio * n.distance:
                good_matches.append(m)
        
        print(f"Good matches found: {len(good_matches)}")
        
        try:
            # Get matching points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            # Get dimensions
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            # Create points for corners
            corners = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            # Get corners of both images
            all_corners = np.vstack((
                transformed_corners,
                np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
            ))
            
            # Find min and max coordinates
            [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel())
            [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel())
            
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
            
            # Combine transformations
            final_matrix = translation.dot(H)
            
            # Create output canvas
            result = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # Warp bottom image
            warped_bottom = cv2.warpPerspective(img2, final_matrix, (output_width, output_height))
            
            # Calculate coordinates for top image
            x_offset = -xmin
            y_offset = -ymin
            
            # Create a mask for blending
            mask = np.zeros((output_height, output_width), dtype=np.uint8)
            
            # Define the region for top image
            top_region = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            y_start = max(0, y_offset)
            y_end = min(output_height, y_offset + h1)
            x_start = max(0, x_offset)
            x_end = min(output_width, x_offset + w1)
            
            # Place top image
            top_region[y_start:y_end, x_start:x_end] = img1[
                max(0, -y_offset):min(h1, output_height-y_offset),
                max(0, -x_offset):min(w1, output_width-x_offset)
            ]
            
            # Create blending mask
            mask[y_start:y_end, x_start:x_end] = 255
            mask_3d = np.stack([mask/255.0]*3, axis=2)
            
            # Blend images
            result = top_region * mask_3d + warped_bottom * (1 - mask_3d)
            
            print(f"Final vertical stitch shape: {result.shape}")
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in vertical stitching: {str(e)}")
            print("Debug info:")
            print(f"Number of good matches: {len(good_matches)}")
            print(f"Source points shape: {src_pts.shape if 'src_pts' in locals() else 'Not created'}")
            print(f"Destination points shape: {dst_pts.shape if 'dst_pts' in locals() else 'Not created'}")
            raise
    def stitch_images(self, images):
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
            print("Debug info:")
            print(f"Number of good matches: {len(good_matches)}")
            print(f"Source points shape: {src_pts.shape if 'src_pts' in locals() else 'Not created'}")
            print(f"Destination points shape: {dst_pts.shape if 'dst_pts' in locals() else 'Not created'}")
            raise
        
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
    
    

def main():
    # Create stitcher instance (using SIFT for better accuracy with text)
    stitcher = ImageStitcher()
    
    # Read images
    folder_path = "data-3"  # Update with your folder path
    images = []
    for i in range(1, 5):
        img_path = os.path.join(folder_path, f"{i}.jpg")
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
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
        cv2.imwrite(folder_path + "_result.jpg", result)
        print("\nResult saved successfully")
        
        cv2.imshow("Stitched Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during stitching: {str(e)}")

if __name__ == "__main__":
    main()