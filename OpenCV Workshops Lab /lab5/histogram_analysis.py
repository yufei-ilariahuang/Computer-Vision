import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
Grayscale Histogram:
• Only one channel, representing the intensity of light from 0 (black) to 255
(white).
• Used to understand image brightness, contrast, and for basic operations
like thresholding.

Color Histogram:
• Three separate histograms, one for each color channel (Blue, Green,
Red).
• Helps analyze the color distribution in an image, used for tasks like object
recognition, color enhancement, and image classification.

'''
def grayscale_histogram(image_path):
    # Load the image in grayscale mode
    img = cv2.imread(image_path, 0)
    '''
    cv2.calcHist(): This OpenCV function computes the histogram of an image.
• [img]: The input image, which is passed as a list because OpenCV expects the image
data in this format.
• [0]: The index of the channel for which the histogram is calculated. Since the image is
grayscale, there's only one channel (0).
• None: This is where a mask would go if you wanted to calculate the histogram for only a
specific region of the image. Here, we use the entire image, so no mask is needed.
• [256]: This represents the number of bins. Since the intensity values of the grayscale
image range from 0 to 255, we use 256 bins to represent all possible values.
• [0, 256]: The range of pixel values we want to consider (0 to 255).

    '''
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

