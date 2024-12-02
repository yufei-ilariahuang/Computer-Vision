import cv2
import numpy as np
import os
from .generative_operations import (
    create_adversarial_noise,
    apply_adversarial_noise,
    apply_diffusion_step,
    denoise_image,
    save_comparison_image
)

def nothing(x):
    """Dummy function for trackbar."""
    pass

def lab11():
    # Create output directory
    if not os.path.exists('lab11'):
        os.makedirs('lab11')
    
    # Load image
    image = cv2.imread('image/w15.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image")
        return
    
    # Resize for consistent processing
    image = cv2.resize(image, (256, 256))
    
    # Create windows
    cv2.namedWindow('Controls')
    cv2.namedWindow('Results')
    
    # Create trackbars
    cv2.createTrackbar('Noise Scale', 'Controls', 25, 100, nothing)
    cv2.createTrackbar('Noise Weight', 'Controls', 10, 100, nothing)
    cv2.createTrackbar('Diffusion Steps', 'Controls', 5, 10, nothing)
    
    while True:
        # Get trackbar values
        noise_scale = cv2.getTrackbarPos('Noise Scale', 'Controls')
        noise_weight = cv2.getTrackbarPos('Noise Weight', 'Controls') / 100.0
        n_steps = cv2.getTrackbarPos('Diffusion Steps', 'Controls')
        
        # Create adversarial image
        noise = create_adversarial_noise(image, noise_scale)
        adversarial = apply_adversarial_noise(image, noise, noise_weight)
        
        # Apply diffusion steps
        diffusion_results = [image]
        current_image = image.copy()
        for i in range(n_steps):
            current_image = apply_diffusion_step(
                current_image, 
                noise_level=0.1 * (i + 1)
            )
            diffusion_results.append(current_image)
        
        # Apply denoising to the most noisy image
        denoised = denoise_image(diffusion_results[-1])
        
        # Display results
        cv2.imshow('Original', image)
        cv2.imshow('Adversarial', adversarial)
        cv2.imshow('Noise', noise)
        
        for i, result in enumerate(diffusion_results[1:]):
            cv2.imshow(f'Diffusion Step {i+1}', result)
        
        cv2.imshow('Denoised', denoised)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Save all images
            cv2.imwrite('lab11/original.jpg', image)
            cv2.imwrite('lab11/adversarial.jpg', adversarial)
            cv2.imwrite('lab11/noise.jpg', noise)
            
            for i, result in enumerate(diffusion_results[1:]):
                cv2.imwrite(f'lab11/diffusion_step_{i+1}.jpg', result)
            
            cv2.imwrite('lab11/denoised.jpg', denoised)
            
            # Create and save comparison
            images = {
                'Original': image,
                'Adversarial': adversarial,
                'Noise': noise,
                'Most Noisy': diffusion_results[-1],
                'Denoised': denoised
            }
            save_comparison_image(images, 'lab11/comparison.jpg')
            
            print("Images saved in lab11/")
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab11()