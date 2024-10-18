import cv2
import matplotlib.pyplot as plt

def display_image(img, window_name="Image"):
    """
    Display an image using matplotlib.
    
    :param img: Image to display
    :param window_name: Title of the plot (default: "Image")
    """
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 2:  # Grayscale image
        plt.imshow(img, cmap='gray')
    else:  # Color image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(window_name)
    plt.axis('off')
    plt.show()

def display_properties(properties):
    """
    Display image properties.
    
    :param properties: Dictionary of image properties
    """
    for key, value in properties.items():
        print(f"{key.capitalize()}: {value}")