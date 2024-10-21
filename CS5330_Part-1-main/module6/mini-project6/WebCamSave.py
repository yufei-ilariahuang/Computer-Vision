# USAGE: python WebCamSave.py -f video_file_name -o out_video.avi

# import the necessary packages
import cv2
import numpy as np
import time
import argparse


# Convert each frame to grayscale to simplify the image data.
def convert_to_grayscale(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame

# Apply Gaussian Blur to reduce noise and make edge detection more accurate.
def apply_gaussian_blur(frame):
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return blurred_frame

# assume lines are white or near white, filter out other colors in the frame
def filter_lines(frame):
    _, frame = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
    return frame

# Define a region of interest in the frame where the lanes are expected to be, typically a trapezoid covering the bottom half of the image.
def set_roi(image):
    height, width = image.shape[:2]
    vertices = np.array([[
        (int(width * 0), height),   # left bottom
        (int(width * 0.45), int(height * 0.3)),  # left top
        (int(width * 0.55), int(height * 0.3)),  # right top
        (int(width * 1), height)    # right bottom
    ]], dtype=np.int32)
    return vertices

# Mask out everything outside ROI.
def mask_image(frame, vertices):
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [vertices], 255)
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image


# Use Canny Edge Detection to identify the edges in the frame, which are likely to represent lane lines.
def apply_canny_edge_detection(frame):
    edges = cv2.Canny(frame, 50, 150)
    return edges


# find hough lines
def find_hough_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=15)
    return lines 

def average_lines(lines):
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                # skip vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            if slope < -0.3:
                left_lines.append((slope, intercept))
            elif slope > 0.3:
                right_lines.append((slope, intercept))
    
    # calculate the average slope and intercept for the left and right lane lines
    left_avg = np.mean(left_lines, axis=0) if len(left_lines) > 0 else None
    right_avg = np.mean(right_lines, axis=0) if len(right_lines) > 0 else None
    
    def make_line(y1, y2, avg):
        if avg is None:
            return None
        slope, intercept = avg
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    
    # generate smooth left and right lane lines based on average slope and intercept
    left_lane = make_line(720, 400, left_avg)
    right_lane = make_line(720, 400, right_avg)
    
    return left_lane, right_lane

def draw_averaged_lines(frame, left_line, right_line):
    # make a copy of the frame
    cp_frame = np.copy(frame)
    if left_line is not None:
        x1, y1, x2, y2 = map(int, left_line)
        cv2.line(cp_frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
    
    if right_line is not None:
        x1, y1, x2, y2 = map(int, right_line)
        cv2.line(cp_frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
    
    return cp_frame

def visualize_roi(frame, vertices):
    cv2.polylines(frame, [vertices], True, (255, 255, 255), 2)
    return frame

# draw edges on original colorful frame
def draw_edges_on_original_frame(frame, edges):
    frame_with_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined_frame = cv2.addWeighted(frame, 0.8, frame_with_edges, 1, 0)
    return combined_frame 

def draw_edges(edges):
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def draw_hough_lines(frame, lines):
    frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    draw_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                # skip vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            if slope < -0.3:
                draw_lines.append(line)
            elif slope > 0.3:
                draw_lines.append(line)

    if draw_lines is not None:
        for line in draw_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # display slope of the line on the frame
            slope = (y2 - y1) / (x2 - x1)
            cv2.putText(frame, f"Slope: {slope:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  
        
    return frame

# Set up argument parser
parser = argparse.ArgumentParser(description="Video file path or camera input")
parser.add_argument("-f", "--file", type=str, help="Path to the video file")
parser.add_argument("-o", "--out", type=str, help="Output video file name")

args = parser.parse_args()

# Check if the file argument is provided, otherwise use the camera
if args.file:
    vs = cv2.VideoCapture(args.file)
else:
    vs = cv2.VideoCapture(0)  # 0 is the default camera

time.sleep(2.0)

# Get the default resolutions
width  = int(vs.get(3))
height = int(vs.get(4))

# Define the codec and create a VideoWriter object
out_filename = args.out
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_filename, fourcc, 20.0, (width, height), True)

mode = "output"
pause = False

# loop over the frames from the video stream
while True:
    if not pause:
        # grab the frame from video stream
        ret, frame = vs.read()
        if not ret:
            break

        # process the frame for lane detection
        gray_frame = convert_to_grayscale(frame)
        gray_gaussian_frame = apply_gaussian_blur(gray_frame)
        white_line_frame = filter_lines(gray_gaussian_frame)

        # mask the image with the ROI   
        vertices = set_roi(white_line_frame)
        masked_frame = mask_image(white_line_frame, vertices)

        # detect edges
        edges = apply_canny_edge_detection(masked_frame)
        edges_frame = draw_edges(edges)

        # detect lines
        lines = find_hough_lines(edges)
        lines_frame = draw_hough_lines(masked_frame, lines)

        # average the lines
        left_line, right_line = average_lines(lines)
        output_frame = draw_averaged_lines(frame, left_line, right_line)

    # output_frame = visualize_roi(masked_frame, vertices)

    if args.out:
        out.write(frame)

    # show the output frame
    if mode == "output":
        cv2.imshow("Frame", output_frame)
    elif mode == "gray":
        cv2.imshow("Gray Frame", gray_frame)
    elif mode == "gaussian":
        cv2.imshow("Gaussian Frame", gray_gaussian_frame)
    elif mode == "white":
        cv2.imshow("White Line Frame", white_line_frame)
    elif mode == "edges":
        cv2.imshow("Edges Frame", edges_frame)
    elif mode == "lines":
        cv2.imshow("Lines Frame", lines_frame)  
    elif mode == "masked":
        cv2.imshow("Masked Frame", masked_frame)
    elif mode == "original":
        cv2.imshow("Original Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("g"):
        mode = "gray"
        cv2.destroyAllWindows()
    elif key == ord("a"):
        mode = "gaussian"
        cv2.destroyAllWindows()
    elif key == ord("w"):
        mode = "white"
        cv2.destroyAllWindows()
    elif key == ord("m"):
        mode = "masked"
        cv2.destroyAllWindows()
    elif key == ord("e"):
        mode = "edges"  
        cv2.destroyAllWindows()
    elif key == ord("l"):
        mode = "lines"
        cv2.destroyAllWindows()
    elif key == ord("o"):
        mode = "output"
        cv2.destroyAllWindows()
    elif key == ord("0"):
        mode = "original"
        cv2.destroyAllWindows()
    elif key == ord("p"):
        pause = not pause
    elif key == ord("q"):
        break


# Release the video capture object
vs.release()
out.release()
cv2.destroyAllWindows()
