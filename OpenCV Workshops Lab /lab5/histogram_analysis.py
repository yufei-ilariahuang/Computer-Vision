import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale_histogram(image_path):
    # Load the image in grayscale mode
    img = cv2.imread(image_path, 0)
    
    # Calculate the histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    # Plot the grayscale histogram
    plt.figure()
    plt.plot(hist)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.show()
    
    return img, hist

def color_histogram(image_path):
    # Load the image in color mode
    img = cv2.imread(image_path)
    
    # Initialize colors for BGR channels
    colors = ('b', 'g', 'r')
    
    plt.figure()
    for i, col in enumerate(colors):
        # Calculate the histogram for each channel
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        
        # Plot the histogram for the current channel
        plt.plot(hist, color=col)
        
    plt.title('Color Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.show()
    
    return img, hist

