import cv2
import time

# Open the video file or capture device (0 is the default camera, or use a video file path)
vs = cv2.VideoCapture(1)
time.sleep(2.0)  # Allow camera to warm up

# Define flags for various operations
crop_enabled = False
resize_enabled = False
blur_enabled = False
box_enabled = False
text_enabled = False
threshold_enabled = False
cartoon_effect_enabled = False

# Define the crop region (start and end points)
crop_region = (100, 100, 400, 400)  # (startX, startY, endX, endY)

# Default resize dimensions
resize_dimensions = (900, 900)

# Function to reset all settings
def reset_settings():
    global crop_enabled, resize_enabled, blur_enabled, box_enabled, text_enabled, threshold_enabled, custom_function_enabled
    crop_enabled = False
    resize_enabled = False
    blur_enabled = False
    box_enabled = False
    text_enabled = False
    threshold_enabled = False
    cartoon_effect_enabled = False

# Function to handle cropping
def crop_frame(frame):
    startX, startY, endX, endY = crop_region
    return frame[startY:endY, startX:endX]

# Function to handle resizing
def resize_frame(frame):
    return cv2.resize(frame, resize_dimensions)

# Function to apply blur
def blur_frame(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

# Function to draw a box
def draw_box(frame):
    cv2.rectangle(frame, (50, 50), (700, 700), (0, 255, 0), 4)
    return frame

# Function to add text
def add_text(frame):
    cv2.putText(frame, "Yufei Huang", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 8)
    return frame

# Function to apply threshold
def apply_threshold(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresholded

# Placeholder for custom function
def cartoon_effect(frame):
    # Convert to gray scale and apply median blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    # Detect edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Apply bilateral filter to smooth colors
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    # Convert to HSV to increase saturation
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    s = cv2.add(s, 50)  # Increase saturation by 50; you can adjust this value as needed
    # Combine edges with color image
    enhanced_hsv = cv2.merge([h, s, v])
    color_saturated = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    color_saturated = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# Loop over the frames from the video stream
while True:
    # Grab the frame from video stream
    ret, frame = vs.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Handle the various key press functionalities
    if crop_enabled:
        frame = crop_frame(frame)

    if resize_enabled:
        frame = resize_frame(frame)

    if blur_enabled:
        frame = blur_frame(frame)

    if box_enabled:
        frame = draw_box(frame)

    if text_enabled:
        frame = add_text(frame)

    if threshold_enabled:
        frame = apply_threshold(frame)

    if cartoon_effect_enabled:
        frame = cartoon_effect(frame)

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Crop the video on 'c' or 'C' key press
    if key == ord("c") or key == ord("C"):
        crop_enabled = not crop_enabled  # Toggle cropping

    # Resize the video on 'r' or 'R' key press
    if key == ord("r") or key == ord("R"):
        resize_enabled = not resize_enabled  # Toggle resizing
        if resize_enabled:
            # Ask the user for new dimensions
            width = int(input("Enter new width: "))
            height = int(input("Enter new height: "))
            resize_dimensions = (width, height)

    # Apply blur on 'b' or 'B' key press
    if key == ord("b") or key == ord("B"):
        blur_enabled = not blur_enabled  # Toggle blur

    # Draw a box on 'a' or 'A' key press
    if key == ord("a") or key == ord("A"):
        box_enabled = not box_enabled  # Toggle drawing box

    # Add text on 't' or 'T' key press
    if key == ord("t") or key == ord("T"):
        text_enabled = not text_enabled  # Toggle text

    # Apply thresholding on 'g' or 'G' key press
    if key == ord("g") or key == ord("G"):
        threshold_enabled = not threshold_enabled  # Toggle thresholding

    # Trigger custom function on 'n' or 'N' key press
    if key == ord("n") or key == ord("N"):
        cartoon_effect_enabled = not cartoon_effect_enabled  # Toggle custom function
        if cartoon_effect_enabled:
            reset_settings()


    # Quit the loop on 'q' key press
    if key == ord("q"):
        break

# Clean up
vs.release()
cv2.destroyAllWindows()
