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
    # Shift the zero-frequency component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    
    # Create a low-pass filter mask
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols, 2), np.uint8)
    # Adjust this value to control the filter strength
    radius = 50  
    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1

    # Apply the mask to the shifted DFT
    dft_shift = dft_shift * mask

    # Shift the zero-frequency component back to the top-left corner
    dft = np.fft.ifftshift(dft_shift)

    # Compute the inverse DFT
    filtered_img = cv2.idft(dft)
    filtered_img = cv2.magnitude(filtered_img[:,:,0], filtered_img[:,:,1])

    # Normalize for display
    filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # plot results
    plot_dft(img, filtered_img)

if __name__ == '__main__':
    main()
