import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    """
    Preprocess image for contour detection.
    
    Args:
        image: Input BGR image
    
    Returns:
        binary: Binary image after preprocessing
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return binary

def find_and_draw_contours(image, mode=cv2.RETR_EXTERNAL):
    """
    Find and draw contours on the image.
    
    Args:
        image: Input BGR image
        mode: Contour retrieval mode
    
    Returns:
        result_image: Image with drawn contours
        contours: List of found contours
    """
    # Make a copy of the image
    result_image = image.copy()
    
    # Preprocess the image
    binary = preprocess_image(image)
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        binary, 
        mode,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Draw all contours
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    return result_image, contours, binary

def analyze_contours(contours):
    """
    Analyze contour properties.
    
    Args:
        contours: List of contours
    
    Returns:
        list: List of dictionaries containing contour properties
    """
    contour_properties = []
    
    for contour in contours:
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = float(w)/h if h != 0 else 0
        
        # Store properties
        properties = {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'bounds': (x, y, w, h)
        }
        
        contour_properties.append(properties)
    
    return contour_properties

def draw_contour_properties(image, contours, properties):
    """
    Draw contour properties on the image.
    
    Args:
        image: Input image
        contours: List of contours
        properties: List of contour properties
    
    Returns:
        Image with properties drawn
    """
    result = image.copy()
    
    for i, (contour, props) in enumerate(zip(contours, properties)):
        # Get bounding rectangle
        x, y, w, h = props['bounds']
        
        # Draw bounding rectangle
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Add text with properties
        text = f"Area: {props['area']:.0f}"
        cv2.putText(result, text, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return result

def save_comparison_image(original, binary, contours, properties, filename):
    """
    Create and save a comparison image showing different stages.
    """
    plt.figure(figsize=(15, 5))
    
    images = [original, binary, contours, properties]
    titles = ['Original', 'Binary', 'Contours', 'Properties']
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i + 1)
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()