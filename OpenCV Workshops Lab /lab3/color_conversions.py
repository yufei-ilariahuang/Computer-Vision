import cv2
'''
use cv2.cvtColor to convert RGB to Grayscale or HSV
    • Grayscale is a color space where each pixel value represents the intensity of light. Instead of having multiple color channels (like RGB), a grayscale image has only one channel, which contains shades of gray ranging from black to white.
        • Grayscale is used in image thresholding, edge/object detection, and facial recognition
    • HSV is a cylindrical color space that describes colors in terms of three components:
        • Hue (H): The color type, represented as a degree on the color wheel (0-360°). In OpenCV,  it's scaled to 0-179.
        • Saturation (S): The intensity or purity of the color (0-255).
        • Value (V): The brightness or lightness of the color (0-255).
        • HSV is used in color filtering, object tracking, and image segmentation
'''
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)