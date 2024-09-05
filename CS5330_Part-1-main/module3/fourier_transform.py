import numpy as np 
import matplotlib.pyplot as plt 
import cv2 

def plot_dft(crop_gray, magnitude_spectrum):
    plt.subplot(121),plt.imshow(crop_gray, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    
def main():
    # read an image as gray image flag=0
    img = cv2.imread('../images/photo.png', 0)
    
    # take discrete fourier transform 
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    
    # plot results
    plot_dft(img, magnitude_spectrum)
    
    
if __name__ == '__main__':
    main()
