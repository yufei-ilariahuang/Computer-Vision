import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

def plot_cv_img(input_image, output_image1, output_image2, output_image3):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=4)

    ax[0].imshow(input_image, cmap='gray')          
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    ax[1].imshow(output_image1, cmap='gray')          
    ax[1].set_title('X Sobel Image')
    ax[1].axis('off')

    ax[2].imshow(output_image2, cmap = 'gray')          
    ax[2].set_title('Y Sobel Image')
    ax[2].axis('off')    

    ax[3].imshow(output_image3, cmap = 'gray')          
    ax[3].set_title('Sobel Image')
    ax[3].axis('off')    

    plt.show()
    

def main():
    # read an image 
    img = cv2.imread('../images/tile.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # sobel 
    x_sobel  = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    y_sobel  = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    # Compute the gradient magnitude
    magnitude = np.sqrt(x_sobel**2 + y_sobel**2)

    # Normalize the magnitude for display
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Do plot
    plot_cv_img(img, x_sobel, y_sobel, magnitude)

if __name__ == '__main__':
    main()
