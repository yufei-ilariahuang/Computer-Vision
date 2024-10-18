import cv2
import os

def save_image(img, file_name):
    """
    Save an image using OpenCV.
    
    :param img: Image to save
    :param file_name: Name of the file to save the image as
    :return: True if successful, False otherwise
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the full file path
    file_path = os.path.join(script_dir, file_name)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    result = cv2.imwrite(file_path, img)
    if result:
        print(f"Image successfully saved to {file_path}")
    else:
        print(f"Error: Could not save image to {file_path}")
    return result