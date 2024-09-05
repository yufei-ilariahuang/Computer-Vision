import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

def plot_cv_img(input_image, output_image):
    """     
    Converts an image from BGR to RGB and plots     
    """

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Corner Image')
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Final Image')
    ax[1].axis('off')

    plt.show()


def harris_corners(input_img):
    """
    computes corners in colored image and plot it.
    """
    # first convert to grayscale with float32 values
    gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # using opencv harris corner implementation
    corners = cv2.cornerHarris(gray,
                               blockSize=2,
                               ksize=5,
                               k=0.04)
    
    # additional thresholding and marking corners for plotting
    input_img[corners>0.01*corners.max()]=[0,0,255]

    return input_img, corners

def main():
    # read an image 
    img = cv2.imread('../images/flower.png')
    
    # compute harris corners and display 
    out, corner = harris_corners(img)
    
    # Do plot
    plot_cv_img(corner, out)

if __name__ == '__main__':
    main()
