# USAGE: python CamTest.py

# import the necessary packages
import cv2
import time
import os

# Open Video Camera
vs = cv2.VideoCapture(0)  # 0 is the default camera
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()

    # Add your code HERE: For example,
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
