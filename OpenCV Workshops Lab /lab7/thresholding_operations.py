import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_simple_threshold(image, threshold_value=127):
    """
    Apply simple thresholding to an image.
    
    Args:
        image: Input grayscale image
        threshold_value: Fixed threshold value (default: 127)
    
    Returns:
        Binary image after thresholding
    """
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

def apply_adaptive_threshold(image, block_size=11, C=2):
    """
    Apply adaptive thresholding using both mean and Gaussian methods.
    
    Args:
        image: Input grayscale image
        block_size: Size of pixel neighborhood (default: 11)
        C: Constant subtracted from mean/weighted sum (default: 2)
    
    Returns:
        Tuple of (mean thresholded image, Gaussian thresholded image)
    """
    adaptive_mean = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, block_size, C
    )
    
    adaptive_gaussian = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )
    
    return adaptive_mean, adaptive_gaussian

def apply_otsu_threshold(image):
    """
    Apply Otsu's binarization method.
    
    Args:
        image: Input grayscale image
    
    Returns:
        Binary image after Otsu's thresholding
    """
    _, otsu = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return otsu

def save_comparison_image(original, simple, adaptive_mean, 
                         adaptive_gaussian, otsu, filename):
    """
    Create and save a comparison image of different thresholding results.
    
    Args:
        original: Original grayscale image
        simple: Simple thresholding result
        adaptive_mean: Adaptive mean thresholding result
        adaptive_gaussian: Adaptive Gaussian thresholding result
        otsu: Otsu's thresholding result
        filename: Output filename
    """
    plt.figure(figsize=(15, 10))
    
    images = [original, simple, adaptive_mean, 
              adaptive_gaussian, otsu]
    titles = ['Original', 'Simple Thresholding', 
              'Adaptive Mean', 'Adaptive Gaussian', 
              "Otsu's Method"]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.close()