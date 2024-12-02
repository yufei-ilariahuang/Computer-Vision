
# Computer Vision Lab Portfolio

This repository documents my journey through a comprehensive Computer Vision course, showcasing a series of labs that explore fundamental to advanced concepts in image processing and analysis using OpenCV and Python.

## Table of Contents

1. [Lab 1: Set Up OpenCV](#lab-1-introduction-to-opencv)
2. [Lab 2: Image Processing Basics](#lab-2-image-processing-basics)
3. [Lab 3: Advanced Image Processing](#lab-3-advanced-image-processing)
4. [Lab 4: Feature Detection and Matching](#lab-4-feature-detection-and-matching)
5. [Lab 5: Image Histograms](#lab-5-image-histograms)
6. [Lab 6: Smoothing and Blurring Techniques](#lab-6-smoothing-and-blurring-techniques)

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

## How to Run
This project includes a main script that allows you to run each lab individually. To run the labs, follow these steps:

Ensure you have set up the environment as described in the Setup and Installation section.
Navigate to the project root directory in your terminal.
Run the main script using Python:
```python
    python main.py
```
You will see a menu with options to run each lab:
CopyComputer Vision Labs
1. Exit
2. Lab 2
3. Lab 3
4. Lab 4
5. Lab 5
6. Lab 6

Enter your choice (1-6):

Enter the number corresponding to the lab you want to run, or enter '1' to exit the program.
The selected lab will run, and you can interact with it as per the lab's instructions.
After each lab finishes, you'll return to the main menu where you can choose to run another lab or exit the program.

Main Script Overview
The main.py script organizes all the labs and provides an easy way to run them. Here's a brief overview of how it works:

It imports each lab's main function (lab2(), lab3(), etc.) from their respective modules.
The main() function presents a menu to the user and runs the selected lab based on user input.
The script uses a while loop to keep presenting the menu until the user chooses to exit.

This structure allows for easy addition of new labs in the future and provides a centralized way to access all the labs in the portfolio.



