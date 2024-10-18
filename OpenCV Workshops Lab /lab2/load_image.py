import cv2

def load_image(image_path, mode=cv2.IMREAD_COLOR):
    """
    Load an image using OpenCV and return it along with its properties.
    
    :param image_path: Path to the image file
    :param mode: Image reading mode (default: cv2.IMREAD_COLOR)
    :return: Tuple containing the image and a dictionary of its properties
    """
    # Load the image
    img = cv2.imread(image_path, mode)
    
    # Check if image is loaded successfully
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, None
    
    # Prepare image properties
    properties = {
        "shape": img.shape,
        "dtype": str(img.dtype),
        "mode": "Color (BGR)" if mode == cv2.IMREAD_COLOR else 
                "Grayscale" if mode == cv2.IMREAD_GRAYSCALE else 
                "Unchanged",
        "first_pixel": img[0,0].tolist()
    }
    
    return img, properties