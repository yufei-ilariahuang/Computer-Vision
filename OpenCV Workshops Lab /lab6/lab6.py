import cv2
from .smoothing_operation import (
    apply_averaging,
    apply_gaussian_blur,
    apply_median_blur,
    apply_bilateral_filter,
    display_image,
    compare_smoothing_methods
)

def lab6():
    # Load an image
    image_path = 'image/w4.jpg'  # Replace with your image path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Display original image
    display_image(image, "Original Image")

    # Apply and display different smoothing methods
    avg_blurred = apply_averaging(image)
    display_image(avg_blurred, "Averaging (Box Filter)")

    gauss_blurred = apply_gaussian_blur(image)
    display_image(gauss_blurred, "Gaussian Blurring")

    median_blurred = apply_median_blur(image)
    display_image(median_blurred, "Median Blurring")

    bilateral_filtered = apply_bilateral_filter(image)
    display_image(bilateral_filtered, "Bilateral Filtering")

    # Compare all methods side by side
    compare_smoothing_methods(image)

if __name__ == "__main__":
    lab6()