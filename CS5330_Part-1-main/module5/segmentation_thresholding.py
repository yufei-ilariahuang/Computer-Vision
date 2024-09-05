import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
image = cv2.imread('../images/tools.png', cv2.IMREAD_GRAYSCALE)

# Global Thresholding
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Adaptive Mean Thresholding
adaptive_mean_thresh = cv2.adaptiveThreshold(image, 255, 
                                             cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

# Adaptive Gaussian Thresholding
adaptive_gaussian_thresh = cv2.adaptiveThreshold(image, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)

# Otsu's Thresholding
_, otsu_thresh = cv2.threshold(image, 0, 255, 
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Plot the results
titles = ['Original Image', 
          'Global Thresholding (v=127)',
          'Adaptive Mean Thresholding', 
          'Adaptive Gaussian Thresholding', 
          "Otsu's Thresholding"]

images = [image, 
          global_thresh, 
          adaptive_mean_thresh, 
          adaptive_gaussian_thresh, 
          otsu_thresh]

plt.figure(figsize=(10, 7))

for i in range(5):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
