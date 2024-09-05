import numpy as np 
import matplotlib.pyplot as plt 
import cv2 


def plot_img(img1, img2):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))          
    ax[0].set_title('Image 1')
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))          
    ax[1].set_title('SIFT Features')
    ax[1].axis('off')

    plt.show()


def main():
    # read an image 
    img = cv2.imread('../images/flower.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Draw keypoints
    output_image = cv2.drawKeypoints(img, keypoints, None, 
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # plot one image image with keypoints
    plot_img(img, output_image)
    

if __name__ == '__main__':
    main()
