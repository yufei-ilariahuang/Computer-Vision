import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Check if frame is captured
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Stack the original and gray images horizontally
    combined = cv2.hconcat([frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])

    # Display the resulting frame
    cv2.imshow('Original and Grayscale', combined)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
