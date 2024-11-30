import cv2
import numpy as np
from .thresholding_operations import (
    apply_simple_threshold,
    apply_adaptive_threshold,
    apply_otsu_threshold,
    save_comparison_image
)
def nothing(x):
    pass
def lab7():
    # Load image in grayscale
    image = cv2.imread('image/w11.webp', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image")
        return

     # Create windows and trackbars
    cv2.namedWindow('Simple Thresholding')
    cv2.createTrackbar('Threshold', 'Simple Thresholding', 127, 255, nothing)
    
    cv2.namedWindow('Adaptive Thresholding')
    cv2.createTrackbar('Block Size', 'Adaptive Thresholding', 11, 99, nothing)
    cv2.createTrackbar('C', 'Adaptive Thresholding', 2, 20, nothing)

    while True:
        # Get current trackbar values
        thresh_val = cv2.getTrackbarPos('Threshold', 'Simple Thresholding')
        block_size = cv2.getTrackbarPos('Block Size', 'Adaptive Thresholding')
        C = cv2.getTrackbarPos('C', 'Adaptive Thresholding')
        
        # Ensure block size is odd
        block_size = block_size if block_size % 2 == 1 else block_size + 1

        # Apply thresholding methods
        simple_thresh = apply_simple_threshold(image, thresh_val)
        adaptive_mean, adaptive_gaussian = apply_adaptive_threshold(image, block_size, C)
        otsu_thresh = apply_otsu_threshold(image)

        # Display results
        cv2.imshow('Original Image', image)
        cv2.imshow('Simple Thresholding', simple_thresh)
        cv2.imshow('Adaptive Thresholding', adaptive_gaussian)
        cv2.imshow('Otsu Thresholding', otsu_thresh)

        # Break loop on 'q' press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Save final results before exiting
            cv2.imwrite('lab7/output_images/simple_threshold.jpg', simple_thresh)
            cv2.imwrite('lab7/output_images/adaptive_mean.jpg', adaptive_mean)
            cv2.imwrite('lab7/output_images/adaptive_gaussian.jpg', adaptive_gaussian)
            cv2.imwrite('lab7/output_images/otsu_threshold.jpg', otsu_thresh)

            # Create and save comparison image
            save_comparison_image(
                image, simple_thresh, adaptive_mean,
                adaptive_gaussian, otsu_thresh,
                'lab7/output_images/comparison.jpg'
            )
            break

    cv2.destroyAllWindows()
    print("Images saved in lab7/output_images/")

if __name__ == "__main__":
    lab7()