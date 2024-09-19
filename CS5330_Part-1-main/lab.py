import cv2

img = cv2.imread("../images/cans.jpg")

#resize
'''

cv2.resize(image, (width, height), fx, fy, interpolation)
• source: Input Image array (Single-channel, 8-bit or floating-point)
• dsize: Size of the output array
• dest: Output array (Similar to the dimensions and type of Input image
array) [optional]
• fx: Scale factor along the horizontal axis [optional]
• fy: Scale factor along the vertical axis [optional]
• interpolation: One of the above interpolation methods [option
'''

#crop

#rotate

"""
To convert an RGB image to Grayscale:
• gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

Grayscale is a color space where each pixel value represents the intensity of
light. 


To convert an RGB image to HSV:
• hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

- Hue (H): The color type, represented as a degree on the color wheel (0-360°). In OpenCV,
it's scaled to 0-179.
• Saturation (S): The intensity or purity of the color (0-255).
• Value (V): The brightness or lightness of the color (0-255)
"""
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("gary", gray_image)
cv2.imshow("hsv", hsv_image)