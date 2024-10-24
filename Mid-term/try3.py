import cv2
import numpy as np
import os

class ImprovedBookStitcher:
    def __init__(self):
        # Initialize SIFT with adjusted parameters
        self.detector = cv2.SIFT_create(
            nfeatures=10000,  # Increased features
            nOctaveLayers=3,
            contrastThreshold=0.02,  # More lenient contrast threshold
            edgeThreshold=20,
            sigma=1.6
        )
        
        # FLANN parameters for matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def stitch_images(self, images):
        """
        Main stitching function with error handling
        """
        if len(images) != 4:
            raise ValueError("Exactly 4 images required")

        # Preprocess images
        processed_images = []
        for img in images:
            if img is None:
                raise ValueError("Invalid image found")
            processed = self.preprocess_image(img)
            processed_images.append(processed)

        # Stitch horizontally first
        try:
            top_row = self.stitch_pair_enhanced(processed_images[0], processed_images[1])
            bottom_row = self.stitch_pair_enhanced(processed_images[2], processed_images[3])
            
            # Stitch vertically
            final_image = self.stitch_vertical_enhanced(top_row, bottom_row)
            
            # Post-process
            return self.post_process(final_image)
        except Exception as e:
            print(f"Error during stitching: {str(e)}")
            return None

    def preprocess_image(self, image):
        """
        Enhance image quality for better feature detection
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to color
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.7, enhanced_color, 0.3, 0)
        
        return result

    def stitch_pair_enhanced(self, img1, img2):
        """
        Enhanced horizontal stitching with more lenient matching
        """
        # Convert to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            raise ValueError("No features detected in one or both images")

        # Match features with more lenient filtering
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        
        for m, n in matches:
            # More lenient ratio test
            if m.distance < 0.85 * n.distance:  # Increased from 0.65
                good_matches.append(m)

        print(f"Number of matches found: {len(good_matches)}")  # Debug info

        # Reduced minimum matches requirement
        if len(good_matches) < 4:  # Minimum required for homography
            print("Warning: Very few matches found")
            return self.fallback_stitch(img1, img2)

        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography with more lenient RANSAC parameters
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("Warning: Could not find homography")
            return self.fallback_stitch(img1, img2)

        # Warp and blend
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Calculate warped size
        corners = np.float32([[0, 0], [0, h2], [w2, 0], [w2, h2]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        [xmin, ymin] = np.int32(warped_corners.min(axis=0).ravel())
        [xmax, ymax] = np.int32(warped_corners.max(axis=0).ravel())

        # Adjust transformation matrix
        translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
        H = translation.dot(H)

        # Warp image
        warped = cv2.warpPerspective(img2, H, (xmax-xmin+1, ymax-ymin+1))

        # Create output canvas
        result = np.zeros((max(h1, ymax-ymin+1), max(w1, xmax-xmin+1), 3), dtype=np.uint8)
        result[0:h1, 0:w1] = img1

        # Simple blending for overlapping regions
        mask = (result != 0) & (warped != 0)
        result = cv2.addWeighted(result, 0.5, warped, 0.5, 0)
        
        return result

    def fallback_stitch(self, img1, img2):
        """
        Simple fallback stitching method when feature matching fails
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Simple side-by-side stitching
        result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        result[:h1, :w1] = img1
        result[:h2, w1:w1+w2] = img2
        
        return result

    def stitch_vertical_enhanced(self, img1, img2):
        """
        Simplified vertical stitching
        """
        if img1 is None or img2 is None:
            raise ValueError("Invalid input images for vertical stitching")

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Create combined image
        result = np.zeros((h1 + h2, max(w1, w2), 3), dtype=np.uint8)
        result[:h1, :w1] = img1
        result[h1:h1+h2, :w2] = img2

        return result

    def post_process(self, image):
        """
        Basic post-processing
        """
        if image is None:
            return None

        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find non-zero area
        coords = cv2.findNonZero(gray)
        if coords is None:
            return image
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop image
        cropped = image[y:y+h, x:x+w]
        
        return cropped

def main():
    # Create stitcher instance
    stitcher = ImprovedBookStitcher()
    
    # Read images
    folder_path = "data-1"  # Update with your folder path
    images = []
    for i in range(1, 5):
        img_path = os.path.join(folder_path, f"{i}.jpg")
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load image: {img_path}")
    
    if len(images) != 4:
        print("Error: Did not find all 4 images")
        return
    
    # Stitch images
    result = stitcher.stitch_images(images)
    
    if result is not None:
        # Save and display result
        output_path = folder_path + ".jpg"
        cv2.imwrite(output_path, result)
        cv2.imshow("Stitched Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed")

