import cv2
import numpy as np

def load_and_display_image(image_path, mode=cv2.IMREAD_COLOR):
    """
    Load an image using OpenCV and display its properties.
    
    :param image_path: Path to the image file
    :param mode: Image reading mode (default: cv2.IMREAD_COLOR)
    """
    # Load the image
    img = cv2.imread(image_path, mode)
    
    # Check if image is loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Display image properties
    print(f"Image shape: {img.shape}")
    print(f"Image data type: {img.dtype}")
    
    # Display different information based on the mode
    if mode == cv2.IMREAD_COLOR:
        print("Image loaded in color mode (BGR)")
        print(f"Pixel at (0,0): {img[0,0]}")  # Show BGR values of first pixel
    elif mode == cv2.IMREAD_GRAYSCALE:
        print("Image loaded in grayscale mode")
        print(f"Pixel at (0,0): {img[0,0]}")  # Show intensity of first pixel
    elif mode == cv2.IMREAD_UNCHANGED:
        if img.shape[-1] == 4:
            print("Image loaded with alpha channel")
            print(f"Pixel at (0,0): {img[0,0]}")  # Show BGRA values of first pixel
        else:
            print("Image loaded unchanged, but no alpha channel detected")
    
    return img

# Example usage
if __name__ == "__main__":
    image_path = "~/CS5330_Part-1-main/images/seal.jpg"  # Replace with your image path
    
    # Load image in color mode (default)
    color_img = load_and_display_image(image_path)
    
    # Load image in grayscale mode
    gray_img = load_and_display_image(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Load image unchanged (with alpha channel if present)
    unchanged_img = load_and_display_image(image_path, cv2.IMREAD_UNCHANGED)

    # Note: To display the images, you would typically use cv2.imshow()
    # But this script focuses on loading and analyzing the images
    # Display the images
    if color_img is not None:
        cv2.imshow("Color Image", color_img)
    if gray_img is not None:
        cv2.imshow("Grayscale Image", gray_img)
    if unchanged_img is not None:
        cv2.imshow("Unchanged Image", unchanged_img)

    # Wait for a key press (optional, keeps the script running)
    cv2.waitKey(0)
    cv2.destroyAllWindows()