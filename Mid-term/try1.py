import cv2
import numpy as np
import os

class BookCoverStitcher:
    def __init__(self, use_sift=True):
        """
        Initialize the stitcher with SIFT (recommended for this case) or ORB
        
        Args:
            use_sift (bool): Use SIFT if True, ORB if False
        """
        if use_sift:
            # SIFT is recommended for this case due to the text and detailed features
            self.detector = cv2.SIFT_create()
            # FLANN parameters for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # ORB configuration if needed
            self.detector = cv2.ORB_create(nfeatures=3000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.use_sift = use_sift

    def stitch_images(self, images):
        """
        Stitch four images arranged in a 2x2 grid
        
        Args:
            images: List of 4 images [top_left, top_right, bottom_left, bottom_right]
        """
        if len(images) != 4:
            raise ValueError("Exactly 4 images required")

        # Step 1: Stitch top row
        top_row = self.stitch_pair(images[0], images[1])
        
        # Step 2: Stitch bottom row
        bottom_row = self.stitch_pair(images[2], images[3])
        
        # Step 3: Stitch rows together
        final_image = self.stitch_vertical(top_row, bottom_row)
        
        # Step 4: Crop and clean up
        return self.post_process(final_image)

    def stitch_pair(self, img1, img2):
        """
        Stitch two images horizontally
        """
        # Detect keypoints and compute descriptors
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        # Match features
        if self.use_sift:
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:  # Stricter ratio for better accuracy
                    good_matches.append(m)
        else:
            matches = self.matcher.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:50]

        # Get matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # Warp images
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate output dimensions
        points = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, H)
        transformed = np.concatenate((transformed, np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)))
        
        [xmin, ymin] = np.int32(transformed.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(transformed.max(axis=0).ravel() + 0.5)
        
        # Translation matrix
        translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
        H = translation.dot(H)
        
        # Warp and blend
        result = cv2.warpPerspective(img2, H, (xmax-xmin, ymax-ymin))
        result[-ymin:h1-ymin, -xmin:w1-xmin] = img1

        return result

    def stitch_vertical(self, img1, img2):
        """
        Stitch two images vertically
        """
        # Similar to stitch_pair but optimized for vertical stitching
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        if self.use_sift:
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        else:
            matches = self.matcher.match(des1, des2)
            good_matches = sorted(matches, key=lambda x: x.distance)[:50]

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        points = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, H)
        transformed = np.concatenate((transformed, np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)))
        
        [xmin, ymin] = np.int32(transformed.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(transformed.max(axis=0).ravel() + 0.5)
        
        translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
        H = translation.dot(H)
        
        result = cv2.warpPerspective(img2, H, (xmax-xmin, ymax-ymin))
        result[-ymin:h1-ymin, -xmin:w1-xmin] = img1

        return result

    def post_process(self, image):
        """
        Post-process the stitched image
        """
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
    stitcher = BookCoverStitcher(use_sift=True)
    
    # Read images
    folder_path = "data-1"  # Update with your folder path
    images = []
    for i in range(1, 5):
        img_path = os.path.join(folder_path, f"{i}.jpg")
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    # Stitch images
    result = stitcher.stitch_images(images)
    
    # Save result
    output_path = folder_path + ".jpg"
    cv2.imwrite(output_path, result)
    
    # Display result
    cv2.imshow("Stitched Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()