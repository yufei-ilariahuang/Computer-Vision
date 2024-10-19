import cv2
import numpy as np
import matplotlib.pyplot as plt
from .histogram_analysis import grayscale_histogram, color_histogram
def lab5():
    image_path = 'image/w3.jpg'  # Replace with your image path
    
    # Grayscale Histogram
    gray_img, gray_hist = grayscale_histogram(image_path)
    cv2.imshow('Grayscale Image', gray_img)
    
    # Color Histogram
    color_img, color_hist = color_histogram(image_path)
    '''
     converting the image from BGR to RGB before displaying it.

This conversion swaps the Red and Blue channels of the image. As a result:

What was originally the Blue channel becomes the Red channel
What was originally the Red channel becomes the Blue channel
The Green channel remains unchanged
    '''
    cv2.imshow('Color Image', cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab5()