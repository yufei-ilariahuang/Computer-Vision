import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_cv_img(input_image, output_image1, output_image2):
    """     
    Converts an image from BGR to RGB and plots     
    """

    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(input_image, cmap='gray')
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(output_image1, cmap='gray')
    ax[1].set_title('Canny Edge')
    ax[1].axis('off')

    ax[2].imshow(output_image2, cmap = 'gray')
    ax[2].set_title('Hough Lines')
    ax[2].axis('off')

    plt.show()


def main():
    # Step 1: Read the image
    img = cv2.imread('./line-images/building.jpg')
    org_img = np.copy(img)

    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply edge detection (Canny)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Step 4: Apply Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, 
                            minLineLength=50, 
                            maxLineGap=10)

    # Step 5: Draw the detected lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        print("No lines detected")

    # Do plot
    plot_cv_img(org_img, edges, img)

if __name__ == '__main__':
    main()

