import cv2

# Initialize the cameras
cap1 = cv2.VideoCapture(0)  # First camera
cap2 = cv2.VideoCapture(1)  # Second camera, change the index if necessary

while True:
    # Capture frame-by-frame from the first camera
    ret1, frame1 = cap1.read()
    # Capture frame-by-frame from the second camera
    ret2, frame2 = cap2.read()

    # Check if frames are captured
    if not ret1 or not ret2:
        break

    # Stack the frames from both cameras horizontally
    combined = cv2.hconcat([frame1, frame2])

    # Display the resulting frames
    cv2.imshow('Camera 1 and Camera 2', combined)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras and close all OpenCV windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
