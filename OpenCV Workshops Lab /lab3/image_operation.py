import cv2
import numpy as np

def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def crop_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    '''
    make the transformation matrix M which will be used for rotating an image
        • Center = center of rotation
        • Angle: Angle is positive for anti-clockwise and negative for clockwise
        • Scale: scaling factor which scales the image
    '''
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    '''
        apply affine transformation to an image, which is a linear mapping method that preserves points, straight lines, and planes
    '''
    return cv2.warpAffine(image, matrix, (w, h))