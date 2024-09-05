import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_cv_img(template_image, output_image):
    """     
    Converts an image from BGR to RGB and plots     
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, gridspec_kw={'width_ratios': [1, 4]})

    ax[0].imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Template')
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Output Image')
    ax[1].axis('off')

    plt.show()

def main():
    # Load the main image and the template image
    image = cv2.imread('../images/livingroom1.png')
    template = cv2.imread('../images/chair.png')

    # Convert the main image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the template
    template_height, template_width = gray_template.shape[:2]

    # Perform template matching
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    
    # Get the locations where the matching result exceeds the threshold
    threshold = 0.53
    locations = np.where(result >= threshold)

    # Draw rectangles around the matched regions
    for point in zip(*locations[::-1]):
        cv2.rectangle(image, point, 
                      (point[0] + template_width, point[1] + template_height),
                      (0, 255, 0), 2)
    # Do plot
    plot_cv_img(template, image)

if __name__ == '__main__':
    main()
