import cv2
import sys
import imutils

img = cv2.imread("../images/flower.png")

if img is None:
    sys.exit("Could not read the image.")

(h, w, d) = img.shape
print("width=", w, "height=", h, "depth=", d)

roi = img[100:400, 50:350]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

#cv2.imshow("Display window", img)
#k = cv2.waitKey(0)

if k == ord("s"):
    cv2.imwrite("flower-copy.png", img)
