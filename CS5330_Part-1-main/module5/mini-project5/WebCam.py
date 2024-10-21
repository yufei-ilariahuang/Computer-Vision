import cv2
import time
import numpy as np
import argparse


class FPSCalculator:
    def __init__(self, update_interval = 1):
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.update_interval = update_interval

    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        if elapsed_time > self.update_interval:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        return self.fps
    

class PanoramaCreator:
    def __init__(self, capture_interval = 0.5):
        self.capture_interval = capture_interval
        self.is_capturing = False
        self.last_captured_time = time.time()
        self.frames = []
        self.result = None
        self.mode = "SIFT"

    def changeMode(self):
        self.mode = "SIFT" if self.mode != "SIFT" else "ORB"

    # handling when a frame is captured from the webcam
    def receiveFrame(self, frame):
        now = time.time()
        if (self.is_capturing is True) and (now - self.last_captured_time) >= self.capture_interval:
            self.last_captured_time = now
            self.frames.append(frame.copy()) # append current frame
    
    def startCapture(self):
        self.is_capturing = True

    def stopCapture(self):
        self.is_capturing = False
        self.processFrames()
    
    # stitching all frames together using STIF or ORB algorithm
    def processFrames(self):
        print("Stitching frames, there are {} frames in total".format(len(self.frames)))
        last_frame = self.frames.pop(0)
        current_frame_index = 1
        while self.frames:
            current_frame = self.frames.pop(0)
            last_frame = self.stitchTwoFrames(last_frame, current_frame, current_frame_index)
            # failed to generate panorama because of no enough matches
            if last_frame is None:
                self.frames.clear()
                break
            current_frame_index += 1
        self.result = last_frame


    def stitchTwoFrames(self, frame1, frame2, current_frame_index):
        # Convert images to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # using ORB or SIFT according to current stitching mode
        algo = cv2.ORB_create() if self.mode == "ORB" else cv2.SIFT_create()

        keypoints1, descriptors1 = algo.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = algo.detectAndCompute(gray2, None)

        # choose different norm type according to current stitching mode 
        norm_type = cv2.NORM_HAMMING if self.mode == "ORB" else cv2.NORM_L2

        # Match features using the BFMatcher
        bf = cv2.BFMatcher(norm_type, crossCheck=True)

        try:
            matches = bf.match(descriptors1, descriptors2)
        except Exception: # no description found in frame 1 or frame 2
            print("Can not stitch frame {} and {}, no keypoints found on one of those two frames".format(current_frame_index, current_frame_index + 1))
            return None


        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # filter out good matches only
        matches = [m for m in matches if m.distance < 0.75 * matches[-1].distance]

        if (len(matches) < 4):
            print("Can not stitch frame {} and {}, no enough good matches")
            return None  # failed to generate panorama

        print("Stitching frame {} and frame {}, {} matches found".format(current_frame_index, current_frame_index + 1, len(matches)))

        # Extract the matched keypoints
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
    
        # Compute the homography matrix using RANSAC
        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Warp the second image to align with the first image
        height1, width1 = frame1.shape[:2]
        height2, width2 = frame2.shape[:2]
        warped_image2 = cv2.warpPerspective(frame2, H, (width1 + width2, height1))

        # Combine the images to create a panorama
        panorama = np.zeros((height1, width1 + width2, 3), dtype=np.uint8)
        panorama[0:height1, 0:width1] = frame1
        panorama[0:height1, 0:width1 + width2] = np.maximum(panorama[0:height1, 0:width1 + width2], warped_image2)
    
        return self.cropBlackArea(panorama)
    

    # remove the black area on the right side of the stitched image using thresholding
    def cropBlackArea(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a binary mask where white pixels represent non-black areas
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours of the non-black areas
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return image  # Return original image if no contours found

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # Crop the image using the bounding box
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image


# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
args = parser.parse_args()

panorama_creator = PanoramaCreator(1)

wait_time = 1
# Check if the file argument is provided, otherwise use the camera
if args.file:
    # launch video file
    cap = cv2.VideoCapture(args.file)
    fps = cap.get(cv2.CAP_PROP_FPS)  # get the actual fps of the video file
    wait_time = int(1000 / fps)  # calculate an appropriate wait time according to the fps of the input video
else:
    # launch camera
    cap = cv2.VideoCapture(1)

time.sleep(2.0)
fps_calc = FPSCalculator()

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    # Check if frame is captured
    if not ret:
        print("Program exit because video is end OR it can not read a frame from the camera, please check your camera connection!")
        break

    # send current frame to panorama creator
    panorama_creator.receiveFrame(frame)

    # update current fps
    fps = fps_calc.update()
    cv2.putText(frame, "FPS = {}".format(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Mode = {}".format(panorama_creator.mode), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if panorama_creator.is_capturing:
        cv2.putText(frame, "Capturing, press 'a' to stop!".format(panorama_creator.mode), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('WebCam/Video Feed', frame)

    key = cv2.waitKey(wait_time) & 0xFF
    # Break the loop on 'q' key press
    if key == ord('q'):
        break
    elif key == ord('s'):
        if not panorama_creator.is_capturing:
            panorama_creator.startCapture()
    elif key == ord('a'):
        if panorama_creator.is_capturing:
            panorama_creator.stopCapture()
            if panorama_creator.result is not None: # if panorama generated correctly
                cv2.imshow("panorama", panorama_creator.result)
                print("Panorama generated successfully, on the preview window, press 'd' to close the preview, press 'f' to save it to 'panorama.jpg', and press q to quit the whole program.")
    elif key == ord('f'):
        if panorama_creator.result is not None: # press f to save current panorama as "panorama.jpg"
            cv2.imwrite("panorama.jpg", panorama_creator.result)
            print("panorama.jpg saved")
    elif key == ord('d'): # press d to close the preview of panorama
        cv2.destroyWindow("panorama")
        panorama_creator.result = None
    elif key == ord('m'):
        panorama_creator.changeMode()
    

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
