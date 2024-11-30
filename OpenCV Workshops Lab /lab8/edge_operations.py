import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel_edge_detection(image):
    """
    Apply Sobel edge detection to an image.
    
    Args:
        image: Input grayscale image
    
    Returns:
        Tuple of (x gradient, y gradient, combined edges)
    """
    # Calculate gradients in x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude
    sobel_combined = cv2.sqrt(sobel_x**2 + sobel_y**2)
    
    # Convert to 8-bit format
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    
    return sobel_x, sobel_y, sobel_combined

def apply_canny_edge_detection(image, threshold1=100, threshold2=200):
    """
    Apply Canny edge detection to an image.
    
    Args:
        image: Input grayscale image
        threshold1: Lower threshold
        threshold2: Upper threshold
    
    Returns:
        Edge detected image
    """
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    return edges

def save_comparison_image(original, sobel_x, sobel_y, sobel_combined, canny, filename):
    """
    Create and save a comparison image of different edge detection results.
    
    Args:
        original: Original grayscale image
        sobel_x: Sobel X gradient
        sobel_y: Sobel Y gradient
        sobel_combined: Combined Sobel edges
        canny: Canny edge detection result
        filename: Output filename
    """
    plt.figure(figsize=(15, 10))
    
    images = [original, sobel_x, sobel_y, sobel_combined, canny]
    titles = ['Original', 'Sobel X', 'Sobel Y', 'Sobel Combined', 'Canny']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()