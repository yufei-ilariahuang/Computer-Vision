import cv2
import numpy as np
from .edge_operations import (
    apply_sobel_edge_detection,
    apply_canny_edge_detection,
    save_comparison_image
)

def nothing(x):
    pass

def lab8():
    # Load image in grayscale
    image = cv2.imread('image/w13.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image")
        return
        
    # Create windows and trackbars for Canny
    cv2.namedWindow('Canny Edge Detection')
    cv2.createTrackbar('Threshold1', 'Canny Edge Detection', 100, 255, nothing)
    cv2.createTrackbar('Threshold2', 'Canny Edge Detection', 200, 255, nothing)

    while True:
        # Get current positions of trackbars
        threshold1 = cv2.getTrackbarPos('Threshold1', 'Canny Edge Detection')
        threshold2 = cv2.getTrackbarPos('Threshold2', 'Canny Edge Detection')
        
        # Apply Sobel edge detection
        sobel_x, sobel_y, sobel_combined = apply_sobel_edge_detection(image)
        
        # Apply Canny edge detection with current thresholds
        canny_edges = apply_canny_edge_detection(image, threshold1, threshold2)
        
        # Display results
        cv2.imshow('Original Image', image)
        cv2.imshow('Sobel X', sobel_x)
        cv2.imshow('Sobel Y', sobel_y)
        cv2.imshow('Sobel Combined', sobel_combined)
        cv2.imshow('Canny Edge Detection', canny_edges)
        
        # Break loop on 'q' press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Save final results
            cv2.imwrite('lab8/Original.jpg', image)
            cv2.imwrite('lab8/Sobel_X.jpg', sobel_x)
            cv2.imwrite('lab8/Sobel_Y.jpg', sobel_y)
            cv2.imwrite('lab8/Sobel_Combined.jpg', sobel_combined)
            cv2.imwrite('lab8/Canny_Edges.jpg', canny_edges)
            
            # Create and save comparison image
            save_comparison_image(
                image, sobel_x, sobel_y, sobel_combined, canny_edges,
                'lab8/comparison.jpg'
            )
            break
            
    cv2.destroyAllWindows()
    print("Images saved in lab8/")

if __name__ == "__main__":
    lab8()