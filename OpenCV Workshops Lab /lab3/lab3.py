import cv2
from .image_operation import resize_image, crop_image, rotate_image
from .color_conversions import to_grayscale, to_hsv
from .drawing_utils import draw_line, draw_rectangle, draw_circle, add_text

def lab3():
    # Load the image
    image_path = "image/w2.jpg"  # Replace with your actual image path
    original = cv2.imread(image_path)

    if original is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Create a copy for drawing operations
    image = original.copy()

    # Initialize variables
    rotation_angle = 0
    drawing_mode = False

    # Dictionary to store all image variations
    images = {
        "Original": original,
        "Drawing": image
    }

    def update_windows():
        for title, img in images.items():
            cv2.imshow(title, img)

    while True:
        update_windows()

        # Wait for a key event
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit
            break
        elif key == ord('r') and not drawing_mode:  # Resize
            images["Resized"] = resize_image(original, 300, 200)
        elif key == ord('c'):  # Crop
            images["Cropped"] = crop_image(original, 100, 100, 200, 200)
        elif key == ord('t') and not drawing_mode:  # Rotate
            rotation_angle = (rotation_angle + 45) % 360
            images["Rotated"] = rotate_image(original, rotation_angle)
        elif key == ord('g'):  # Grayscale
            images["Grayscale"] = to_grayscale(original)
        elif key == ord('h'):  # HSV
            images["HSV"] = to_hsv(original)
        elif key == ord('d'):  # Enter drawing mode
            drawing_mode = True
            images["Drawing"] = original.copy()
            print("Entered drawing mode. Press 'x' to exit drawing mode.")
        
        # Drawing mode operations
        if drawing_mode:
            if key == ord('l'):  # Line
                draw_line(images["Drawing"], (0, 0), (100, 100), (255, 0, 0), 2)
            elif key == ord('r'):  # Rectangle
                draw_rectangle(images["Drawing"], (50, 50), (150, 150), (0, 255, 0), 2)
            elif key == ord('i'):  # Circle
                draw_circle(images["Drawing"], (100, 100), 50, (0, 0, 255), 2)
            elif key == ord('t'):  # Text
                add_text(images["Drawing"], "OpenCV", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif key == ord('x'):  # Exit drawing mode
                drawing_mode = False
                print("Exited drawing mode.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab3()