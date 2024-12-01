import cv2
import numpy as np
from .contour_operations import (
    find_and_draw_contours,
    analyze_contours,
    draw_contour_properties,
    save_comparison_image
)

def nothing(x):
    pass

def lab9():
    # Load image
    image = cv2.imread('image/w14.webp')
    if image is None:
        print("Error: Could not load image")
        return
    
    # Create window and trackbar for threshold
    cv2.namedWindow('Contour Detection')
    cv2.createTrackbar('Threshold', 'Contour Detection', 127, 255, nothing)
    cv2.createTrackbar('Mode', 'Contour Detection', 0, 2, nothing)
    
    while True: 
        # Get current trackbar values
        threshold = cv2.getTrackbarPos('Threshold', 'Contour Detection')
        mode = cv2.getTrackbarPos('Mode', 'Contour Detection')
        
        # Map mode value to OpenCV contour modes
        retrieval_modes = [
            cv2.RETR_EXTERNAL,
            cv2.RETR_LIST,
            cv2.RETR_TREE
        ]
        current_mode = retrieval_modes[mode]
        
        # Find and draw contours
        contour_image, contours, binary = find_and_draw_contours(image, current_mode)
        
        # Analyze contours
        properties = analyze_contours(contours)
        
        # Draw properties
        properties_image = draw_contour_properties(image, contours, properties)
        
        # Display results
        cv2.imshow('Original', image)
        cv2.imshow('Binary', binary)
        cv2.imshow('Contour Detection', contour_image)
        cv2.imshow('Properties', properties_image)
        
        # Break loop on 'q' press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Save final results
            cv2.imwrite('lab9/Original.jpg', image)
            cv2.imwrite('lab9/Binary.jpg', binary)
            cv2.imwrite('lab9/Contours.jpg', contour_image)
            cv2.imwrite('lab9/Properties.jpg', properties_image)
            
            # Create and save comparison image
            save_comparison_image(
                image, binary, contour_image, properties_image,
                'lab9/comparison.jpg'
            )
            break
    
    cv2.destroyAllWindows()
    print("Images saved in lab9/")

if __name__ == "__main__":
    lab9()