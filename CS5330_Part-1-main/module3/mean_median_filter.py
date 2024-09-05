import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

def plot_cv_img(input_image, output_image1, output_image2):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))          
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(output_image1, cv2.COLOR_BGR2RGB))          
    ax[1].set_title('Mean Filter (5,5)')
    ax[1].axis('off')

    ax[2].imshow(cv2.cvtColor(output_image2, cv2.COLOR_BGR2RGB))          
    ax[2].set_title('Median Filter (5,5)')
    ax[2].axis('off')    

    plt.show()
    

def main():
    # read an image 
    img = cv2.imread('../images/flower.png')

    mean = cv2.blur(img,(5,5))

    median = cv2.medianBlur(img,5)
    
    # Do plot
    plot_cv_img(img, mean, median)

if __name__ == '__main__':
    main()
