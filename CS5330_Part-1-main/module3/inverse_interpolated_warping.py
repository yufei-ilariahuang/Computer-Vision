import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_cv_img(input_image, output_image, final_image):
    """     
    Converts an image from BGR to RGB and plots     
    """

    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Forward Warped Image')
    ax[1].axis('off')

    ax[2].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Inverse Warped Image')
    ax[2].axis('off')
    plt.show()

def forward_warp(image, M):
    h, w = image.shape[:2]
    output = np.zeros_like(image)
    
    # Create coordinate grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()])

    # Apply transformation
    new_coords = M @ coords
    new_coords = new_coords / new_coords[2, :]  # Normalize by the third row
    
    # Reshape to integer pixel locations
    new_x = new_coords[0, :].astype(int).reshape(h, w)
    new_y = new_coords[1, :].astype(int).reshape(h, w)
    
    # Mask to keep coordinates within image bounds
    mask = (new_x >= 0) & (new_x < w) & (new_y >= 0) & (new_y < h)
    
    # Warp the image
    output[new_y[mask], new_x[mask]] = image[y[mask], x[mask]]
    
    return output

def bilinear_interpolate(image, x, y):
    # Fill up here using LLM

def inverse_warp_with_interpolation(image, M):
    # Fill up here using LLM


def main():
    # read an image 
    img = cv2.imread('../images/flower.png')

    # Define affine transformation matrix (example: rotation + translation)
    angle = 30  # Rotation angle in degrees
    scale = 1.2
    tx, ty = 100, 50  # Translation values
    center = (img.shape[1] // 2, img.shape[0] // 2)
    
    # Create affine transformation matrix
    M_rot = cv2.getRotationMatrix2D(center, angle, scale)
    # Convert to 3x3 matrix for homogeneous coordinates
    M = np.vstack([M_rot, [0, 0, 1]])  
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply forward warping
    forward_warped_image = forward_warp(img, M)

    # Apply inverse warping with interpolation using the original image as input
    # Fill up here using LLM
    
    # Do plot
    plot_cv_img(img, forward_warped_image, final_image)

if __name__ == '__main__':
    main()
