import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_averaging(image, kernel_size=(5, 5)):
    return cv2.blur(image, kernel_size)

def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_median_blur(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def display_image(image, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def compare_smoothing_methods(image):
    methods = [
        ("Original", lambda img: img),
        ("Averaging (Box Filter)", apply_averaging),
        ("Gaussian Blurring", apply_gaussian_blur),
        ("Median Blurring", apply_median_blur),
        ("Bilateral Filtering", apply_bilateral_filter)
    ]

    plt.figure(figsize=(20, 10))
    for i, (title, method) in enumerate(methods):
        smoothed = method(image)
        plt.subplot(2, 3, i+1)
        plt.imshow(cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()