# USAGE
# python imagestill.py

# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os

vs = cv2.VideoCapture(-1)
#vs.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
#vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)

time.sleep(1.0)
i = 0                           # Image sequence

# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # if the `s` key was pressed, save current image
    if key == ord("s"):
        filename = "test" + str(i) + ".jpg"
        cv2.imwrite("./data/" + filename, frame)
        print(filename + " is saved.")
        i += 1

cv2.destroyAllWindows()
vs.release()
