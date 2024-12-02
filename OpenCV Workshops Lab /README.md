
# Computer Vision Lab Portfolio

This repository documents my journey through a comprehensive Computer Vision course, showcasing a series of labs that explore fundamental to advanced concepts in image processing and analysis using OpenCV and Python.

## Table of Contents

1. [Lab 1: Set Up OpenCV](#lab-1-introduction-to-opencv)
2. [Lab 2: Image Processing Basics](#lab-2-image-processing-basics)
3. [Lab 3: Advanced Image Processing](#lab-3-advanced-image-processing)
4. [Lab 4: Feature Detection and Matching](#lab-4-feature-detection-and-matching)
5. [Lab 5: Image Histograms](#lab-5-image-histograms)
6. [Lab 6: Smoothing and Blurring Techniques](#lab-6-smoothing-and-blurring-techniques)
7. [Lab 7: Image Thresholding](#lab-7-image-thresholding)
8. [Lab 8: Edge Detection](#lab-8-edge-detection)
9. [Lab 9: Contour Detection](#lab-9-contour-detection)
10. [Lab 10: Optical Flow](#lab-10-optical-flow)
11. [Lab 11: Generative AI](#lab-11-generative-ai)

## Project Structure
```
OpenCV Workshops Lab/
├── image/                  # Shared image resources
├── lab2/ - lab11/         # Individual lab directories
│   ├── README.md          # Lab-specific documentation
│   ├── lab*.py            # Main lab script
│   ├── *_operations.py    # Supporting operations
│   └── *.jpg             # Output images
├── main.py                # Central execution script
└── opencv_env.yml         # Environment configuration
```

## Lab 1: Introduction to OpenCV

This lab covers the setup and installation of OpenCV, providing a foundation for the course. Here I use Anaconda for virtual envirnment for this lab

## Setup and Installation

```bash
conda env create -f environment.yml
conda activate cv_lab
```


## Lab 2: Image Processing Basics

Focusing on fundamental image processing techniques, this lab explores:
- Image loading and display
- Color space conversions
- Basic image manipulations (darkening, grayscale conversion)

Key learnings include understanding different color spaces and implementing basic image transformations.

## Lab 3: Advanced Image Processing

This lab builds on the basics, introducing more complex operations:
- Real-time image resizing and cropping
- Image rotation
- Interactive drawing on images
- Color space conversions (Grayscale and HSV)

The lab implements an interactive program, allowing real-time application of various image processing techniques.

## Lab 4: Feature Detection and Matching

Concentrating on feature detection and matching techniques, this lab covers:
- Harris Corner Detection
- SIFT (Scale-Invariant Feature Transform)
- FLANN (Fast Library for Approximate Nearest Neighbors) matching

The lab applies these techniques to dog images, demonstrating practical applications in object recognition and image matching.

## Lab 5: Image Histograms

This lab delves into the creation and analysis of image histograms:
- Grayscale histograms
- Color histograms
- Applications in image analysis and enhancement

Students gain insights into using histograms for tasks such as brightness analysis, color composition understanding, and image comparison.

## Lab 6: Smoothing and Blurring Techniques

The final lab explores various smoothing and blurring methods:
- Averaging (Box Filter)
- Gaussian Blurring
- Median Blurring
- Bilateral Filtering

These techniques are crucial for noise reduction and image preprocessing in computer vision tasks.



## Lab 7: Image Thresholding

This lab explores various thresholding techniques:
- Simple Thresholding
- Adaptive Thresholding
- Otsu's Thresholding

Key implementations demonstrate how different thresholding methods affect image segmentation and binary image creation.

## Lab 8: Edge Detection

This lab focuses on edge detection algorithms:
- Sobel Edge Detection (X and Y directions)
- Combined Sobel Edge Detection
- Canny Edge Detection

Students learn to identify and extract edges from images using different approaches and parameters.

## Lab 9: Contour Detection

This lab covers contour detection and analysis:
- Binary image preprocessing
- Contour detection methods
- Property analysis (area, perimeter)
- Contour visualization techniques

Practical applications include object detection and shape analysis.

## Lab 10: Optical Flow

This lab explores motion detection using optical flow:
- Camera setup and operation
- Sparse optical flow (Lucas-Kanade)
- Dense optical flow (Farneback)
- Real-time motion tracking

The lab implements both sparse and dense optical flow methods for motion analysis.

## Lab 11: Generative AI

This lab introduces generative AI concepts through:
- Adversarial image generation
- Forward diffusion process
- Basic denoising operations
- Interactive parameter control

Students experiment with noise manipulation and image recovery techniques.


## How to Run

1. Run the main script:

```python
python main.py
```

2. Select lab number from menu (1-11) or 'q' to exit.

You will see a menu with options to run each lab:
CopyComputer Vision Labs
1. Exit
2. Lab 2
3. Lab 3
4. Lab 4
5. Lab 5
6. Lab 6
...


Each lab contains:
- Main implementation file
- Supporting operation modules
- Comprehensive README
- Output images and comparisons



Enter the number corresponding to the lab you want to run, or enter '1' to exit the program.
The selected lab will run, and you can interact with it as per the lab's instructions.
After each lab finishes, you'll return to the main menu where you can choose to run another lab or exit the program.



