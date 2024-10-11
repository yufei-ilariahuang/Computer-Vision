import cv2
import time
import numpy as np

class FrameProcessor:
    # initialize instance variables for different warping mode
    def __init__(self):
        # parameters for image translation
        self.translation = False
        self.translation_x = 0
        self.translation_y = 0
        
        # parameters for image rotation
        self.rotation = False
        self.rotation_degree = 0

        # parameters for image scaling
        self.scaling = False
        self.scaling_factor = 1

        # parameters for image perspective transformation
        self.perspective_transformation = False

    # reset all parameters to default values
    def reset(self):
        self.__init__()

    def leftBtnPressed(self):
        # if in translation mode, image is going to be moved left by 50 pixel
        if self.translation:
            self.translation_x -= 50

        # if in rotation mode, image will be rotate counterclockwise by 1 degree
        if self.rotation:
            self.rotation_degree += 1
    
    def rightBtnPressed(self):
        # if in translation mode, image is going to be moved right by 50 pixel
        if self.translation:
            self.translation_x += 50

        # if in rotation mode, image will be rotate clockwise by 1 degree
        if self.rotation:
            self.rotation_degree -= 1

    def upBtnPressed(self):
        # if in translation mode, image is going to be moved up by 50 pixel
        if self.translation:
            self.translation_y -= 50

        # if in scaling mode, image will be scaled up by 1%
        if self.scaling:
            self.scaling_factor += 0.01


    def downBtnPressed(self):
        # if in translation mode, image is going to be moved down by 50 pixel
        if self.translation:
            self.translation_y += 50

        # if in scaling mode, image will be scaled down by 1%
        if self.scaling and self.scaling_factor > 0.01:
            self.scaling_factor -= 0.01

    def changeMode(self, key):
        if key == ord("t"):  # set to translation mode
            temp = not self.translation
            self.reset()
            self.translation = temp
        elif key == ord("r"):  # set to rotation mode
            temp = not self.rotation
            self.reset()
            self.rotation = temp
        elif key == ord("s"):  # set to scaling mode
            temp = not self.scaling
            self.reset()
            self.scaling = temp
        elif key == ord("p"):  # set to perspective transformation mode
            temp = not self.perspective_transformation
            self.reset()
            self.perspective_transformation = temp
        elif key == ord("y"):  # letter y got mapped as up button
            self.upBtnPressed()
        elif key == ord("h"):  # letter h got mapped as down button
            self.downBtnPressed()
        elif key == ord("g"):  # letter g got mapped as left button 
            self.leftBtnPressed()
        elif key == ord("j"):  # letter j got mapped as right button
            self.rightBtnPressed()
            
    # Apply active warping function to current frame
    def getProcessedFrame(self, frame):
        processed_frame = frame.copy()
        height, width = processed_frame.shape[:2]

        if self.translation:
            shift_matrix = np.float32([[1, 0, self.translation_x], [0, 1, self.translation_y]])
            processed_frame = cv2.warpAffine(processed_frame, shift_matrix, (width, height))

        if self.rotation:
            rotation_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), self.rotation_degree, 1.0)
            processed_frame = cv2.warpAffine(processed_frame, rotation_matrix, (width, height))

        if self.scaling:
            newWidth = int(width * self.scaling_factor)
            newHeight = int(height * self.scaling_factor)
            halfWidthDiff = abs(width - newWidth) // 2
            halfHeightDiff = abs(height - newHeight) // 2
            scaled_frame = cv2.resize(processed_frame, (newWidth, newHeight), interpolation=cv2.INTER_LINEAR)
            if self.scaling_factor > 1:
                # do cropping since the scaled frame is larger than original frame
                processed_frame = scaled_frame[halfHeightDiff: halfHeightDiff + height, halfWidthDiff: halfWidthDiff + width]
            elif self.scaling_factor < 1:
                topPadding = halfHeightDiff
                bottomPadding = height - newHeight - topPadding
                leftPadding = halfWidthDiff
                rightPadding = width - newWidth - leftPadding
                # do paddding since the scaled frame is smaller than original frame
                processed_frame = cv2.copyMakeBorder(scaled_frame, topPadding, bottomPadding, leftPadding, rightPadding, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if self.perspective_transformation:
            # fixed-perspective transformation angle
            point1 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
            point2 = np.float32([[100, 100], [0, height - 100], [width - 100, height - 100], [width - 100, 0]])
            transformation_matrix = cv2.getPerspectiveTransform(point1, point2)

            processed_frame = cv2.warpPerspective(processed_frame, transformation_matrix, (width, height))

        # put a text indicator of the mode
        modeStr = "Original"
        if self.translation:
            modeStr = "Translation"
        elif self.rotation:
            modeStr = "Rotation"
        elif self.scaling:
            modeStr = "Scaling"
        elif self.perspective_transformation:
            modeStr = "Perspective Transformation" 
        cv2.putText(processed_frame, modeStr, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return processed_frame

# Main Function
# Initialize the camera
cap = cv2.VideoCapture(1)  # Change the index if you have multiple cameras
frameProcessor = FrameProcessor()
start_time = time.time()

while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame is captured
    if not ret:
        break

    # Apply active warping to the frame, then compute and display FPS on Terminal/Console
    processed_frame = frameProcessor.getProcessedFrame(frame)
    combined = cv2.hconcat([frame, processed_frame])
    
    # current frame process finished time
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(combined, "FPS = {}".format(fps), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    start_time = end_time

    # Display the resulting frame
    cv2.imshow('Original and Warpped Frame', combined)

    key = cv2.waitKey(1) & 0xFF
    # Break the loop on 'q' key press
    if key == ord('q'):
        break
    # Toggle/Change Frame Processor Mode
    else:
        frameProcessor.changeMode(key)
    

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
