import cv2
import numpy as np
import matplotlib.pyplot as plt
from .load_image import load_image
from .display_image import display_image, display_properties
from .save_image import save_image

def darker_opencv(image):
    return cv2.convertScaleAbs(image, alpha=0.5, beta=0)

def grayscale_opencv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compare_methods(image_path):
    # Load image
    cv_img = cv2.imread(image_path)
    
    # Apply methods
    darker_img = darker_opencv(cv_img)
    gray_img = grayscale_opencv(cv_img)
    hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    # Save results
    cv2.imwrite("original.jpg", cv_img)
    cv2.imwrite("darker_opencv.jpg", darker_img)
    cv2.imwrite("gray_opencv.jpg", gray_img)
    cv2.imwrite("hsv_opencv.jpg", hsv_img)

    # Display results
    plt.figure(figsize=(20, 10))
    images = [
        ("Original", cv_img),
        ("Darker (OpenCV)", darker_img),
        ("Grayscale (OpenCV)", cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)),
        ("HSV (OpenCV)", hsv_img)
    ]

    for i, (title, img) in enumerate(images):
        plt.subplot(2, 2, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("comparison_results.png")
    plt.show()

def lab2():
    image_path = "image/w1.jpg"  # Replace with your image path
    
    # Load image in different modes
    color_img, color_props = load_image(image_path)
    gray_img, gray_props = load_image(image_path, cv2.IMREAD_GRAYSCALE)
    unchanged_img, unchanged_props = load_image(image_path, cv2.IMREAD_UNCHANGED)
    
    # Display images and properties
    if color_img is not None:
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        plt.title("Color Image")
        plt.axis('off')
        print("Color Image Properties:")
        display_properties(color_props)
    
    if gray_img is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(gray_img, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        print("\nGrayscale Image Properties:")
        display_properties(gray_props)
        save_image(gray_img, "gray_img.jpg")
    
    if unchanged_img is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(unchanged_img, cv2.COLOR_BGR2RGB))
        plt.title("Unchanged Image")
        plt.axis('off')
        print("\nUnchanged Image Properties:")
        display_properties(unchanged_props)

    # Display the saved grayscale image
    saved_gray_img = cv2.imread("gray_img.jpg", cv2.IMREAD_GRAYSCALE)
    if saved_gray_img is not None:
        plt.subplot(1, 4, 4)
        plt.imshow(saved_gray_img, cmap='gray')
        plt.title("Saved Grayscale Image")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    # Compare methods
    compare_methods(image_path)

if __name__ == "__main__":
    lab2()