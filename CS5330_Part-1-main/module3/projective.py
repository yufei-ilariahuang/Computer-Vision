import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

def plot_cv_img(input_image, output_image):     
    """     
    Converts an image from BGR to RGB and plots     
    """   

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))          
    ax[0].set_title('Input Image')
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))          
    ax[1].set_title('Output Image')
    ax[1].axis('off')    
    
    plt.show()
    

def main():
    # read an image 
    img = cv2.imread('../images/flower.png')

    # Define points for projective transformation
    pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250], [250, 200]])

    # Get projective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply projective transformation
    projective_image = cv2.warpPerspective(img, M, 
                                          (img.shape[1], img.shape[0]))

    # Do plot
    plot_cv_img(img, projective_image)

if __name__ == '__main__':
    main()
