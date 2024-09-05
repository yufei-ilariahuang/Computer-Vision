# USAGE: python CamTestfile.py -f video_file_name

# import the necessary packages
import cv2
import numpy as np
import time
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
args = parser.parse_args()

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        break

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
