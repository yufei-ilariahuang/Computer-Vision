import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

def plot_dft(crop_gray, filtered):
    plt.subplot(121),plt.imshow(crop_gray, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(filtered, cmap = 'gray')
    plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
def main():
    # read an image
    img = cv2.imread('../images/photo.png')

    # Apply bilateral filter
    filtered_img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # plot results
    plot_dft(img, filtered_img)

if __name__ == '__main__':
    main()
