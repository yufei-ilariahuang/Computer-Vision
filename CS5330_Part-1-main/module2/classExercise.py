import cv2
#import library
import numpy as np

img = cv2.imread('../images/burrito.jpg')
# cv2.IMREAD_COLOR – It specifies to load a color image. Any transparency of
# image will be neglected. It is the default flag. Alternatively, we can pass integer
# value 1 for this flag.
# • cv2.IMREAD_GRAYSCALE – It specifies to load an image in grayscale mode.
# Alternatively, we can pass integer value 0 for this flag.
# • cv2.IMREAD_UNCHANGED – It specifies to load an image as such including
# alpha channel. Alternatively, we can pass integer value -1 for this flag.
gray_img = cv2.imread('../images/burrito.jpg', cv2.IMREAD_GRAYSCALE)


#Method to display an image in a window
cv2.imshow("image", img)

# Allows users to display a window for given milliseconds or until any key is
# pressed.
# • Takes time in milliseconds as a parameter and waits for the given time to
# destroy the window, if 0 is passed in the argument it waits till any key is
# pressed.
cv2.waitKey(2000)

# allows users to destroy or close all windows at any time after exiting the
# script
cv2.destroyAllWindows()