import cv2
import sys
import imutils
# Load the image
img = cv2.imread("../../images/flower.png")

if img is None:
    sys.exit("Could not read the image.")
# Print image properties
(h, w, d) = img.shape
print("width=", w, "height=", h, "depth=", d)

# 1. Show Region of Interest (ROI)
startY, endY, startX, endX = 60, 160, 320, 420
roi = img[startY:endY, startX:endX]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

# 2. Resize Image to (200, 200) while maintaining the aspect ratio
# Calculate the aspect ratio and resize
aspect_ratio = w / h
new_width = 200
new_height = int(new_width / aspect_ratio)
resized_img = cv2.resize(img, (new_width, new_height))
cv2.imshow("Resized Image", resized_img)
cv2.waitKey(0)



# 3. Rotate Image 45 degrees clockwise
rotated_img = imutils.rotate(img, -45)  # Negative angle for clockwise
cv2.imshow("Rotated Image", rotated_img)
cv2.waitKey(0)

# 4. Smooth Image using GaussianBlur
smoothed_img = cv2.GaussianBlur(img, (15, 15), 0)  # Kernel size (15, 15)
cv2.imshow("Smoothed Image", smoothed_img)
cv2.waitKey(0)


# 5. Drawing: Rectangle, Circle, and Line
drawing_img = img.copy()
# Draw rectangle
cv2.rectangle(drawing_img, (50, 50), (150, 150), (0, 255, 0), 2)  # Green rectangle
# Draw circle
cv2.circle(drawing_img, (300, 300), 50, (255, 0, 0), 3)  # Blue circle
# Draw line
cv2.line(drawing_img, (100, 100), (400, 400), (0, 0, 255), 5)  # Red line
cv2.imshow("Drawing on Image", drawing_img)
cv2.waitKey(0)

# 6. Add Text: Insert "Your Name" onto the image
text_img = img.copy()
cv2.putText(text_img, "Yufei Huang", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow("Text on Image", text_img)
cv2.waitKey(0)

# 7. Convert to Grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray_img)
cv2.waitKey(0)

# 8. Edge Detection using Canny method
edges = cv2.Canny(gray_img, 100, 200)
cv2.imshow("Edge Detection", edges)
cv2.waitKey(0)

# 9. Thresholding
_, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresholded Image", threshold_img)
cv2.waitKey(0)

# 10. Detect and Draw Contours
contours_img = img.copy()
contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 2)  # Draw contours in green
cv2.imshow("Contours", contours_img)
cv2.waitKey(0)
#cv2.imshow("Display window", img)
#k = cv2.waitKey(0)


# Save the modified images
cv2.imwrite("flower_roi.png", roi)
cv2.imwrite("flower_resized.png", resized_img)
cv2.imwrite("flower_rotated.png", rotated_img)
cv2.imwrite("flower_smoothed.png", smoothed_img)
cv2.imwrite("flower_drawing.png", drawing_img)
cv2.imwrite("flower_text.png", text_img)
cv2.imwrite("flower_grayscale.png", gray_img)
cv2.imwrite("flower_edges.png", edges)
cv2.imwrite("flower_threshold.png", threshold_img)
cv2.imwrite("flower_contours.png", contours_img)

# Close all windows
cv2.destroyAllWindows()