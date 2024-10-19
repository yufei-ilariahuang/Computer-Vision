
# Lab 3: Advanced Image Processing with OpenCV

## Introduction
This lab focuses on advanced image processing techniques using OpenCV. We explore basic image operations, color space conversions, and drawing shapes and text on images. The lab implements an interactive program that allows users to perform various image processing operations in real-time.

## Features
- Image loading and display
- Real-time image resizing
- Image cropping
- Image rotation
- Color space conversions (Grayscale and HSV)
- Interactive drawing mode (lines, rectangles, circles, and text)
- Save functionality for processed images

## Code Structure
The lab consists of several Python files:
- `lab3.py`: Main script containing the `lab3()` function
- `image_operation.py`: Contains functions for resizing, cropping, and rotating images
- `color_conversions.py`: Contains functions for color space conversions
- `drawing_utils.py`: Contains functions for drawing shapes and text on images

## Usage
Run the `lab3()` function to start the interactive image processing program. Use keyboard inputs to perform different operations:

- 'q': Quit the program
- 'r': Resize the image
- 'c': Crop the image
- 't': Rotate the image
- 'g': Convert to grayscale
- 'h': Convert to HSV
- 'd': Enter drawing mode
  - 'l': Draw a line
  - 'r': Draw a rectangle
  - 'i': Draw a circle
  - 't': Add text
  - 'x': Exit drawing mode

## Key Operations

### Resizing Images
![resized](https://github.com/user-attachments/assets/ec31e764-5c32-4756-9c25-fd28eefb0c5b)

```python
resized = resize_image(original, 300, 200)
```

### Cropping Images
![cropped](https://github.com/user-attachments/assets/b471de9d-76ed-44c2-b6e9-8bb0a9a5cb1d)

```python
cropped = crop_image(original, 100, 100, 200, 200)
```

### Rotating Images
![rotated](https://github.com/user-attachments/assets/4399152f-2f74-4ab1-a3a4-e5f78a872fff)

```python
rotated = rotate_image(original, angle)
```

### Color Space Conversions
![grayscale](https://github.com/user-attachments/assets/0a93a69a-237e-4942-925f-8f906d3216ee)

```python
gray = to_grayscale(original)
hsv = to_hsv(original)
```

### Drawing Shapes and Text
![drawing](https://github.com/user-attachments/assets/6ed9191f-f11b-404f-adbd-b0d7f2927693)

```python
draw_line(image, (0, 0), (100, 100), (255, 0, 0), 2)
draw_rectangle(image, (50, 50), (150, 150), (0, 255, 0), 2)
draw_circle(image, (100, 100), 50, (0, 0, 255), 2)
add_text(image, "OpenCV", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
```

## Observations and Explanations
1. The program uses separate windows for each image operation, allowing for easy comparison.
2. Real-time updates provide immediate feedback on the effects of each operation.
3. The drawing mode demonstrates how to interactively modify images using OpenCV.
4. Color space conversions show how images can be represented in different formats for various processing tasks.

## Additional Explorations
1. Implement more advanced image transformations (e.g., affine transformations).
2. Add image filtering operations (e.g., blur, edge detection).
3. Implement a more sophisticated drawing interface with mouse interactions.
4. Explore image segmentation techniques using thresholding or clustering algorithms.

## Conclusion
This lab provides hands-on experience with advanced image processing techniques using OpenCV. It demonstrates how to create an interactive program for image manipulation, combining various OpenCV functions into a cohesive application. These skills form the foundation for more complex computer vision tasks and can be extended to real-world applications in fields such as medical imaging, document processing, or computer graphics.
