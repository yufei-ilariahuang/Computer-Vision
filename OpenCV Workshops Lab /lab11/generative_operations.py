import cv2
import numpy as np

def create_adversarial_noise(image, noise_scale=25.0):
    """
    Create random noise for adversarial image generation.
    
    Args:
        image: Input image
        noise_scale: Standard deviation of the noise
    
    Returns:
        Generated noise array
    """
    return np.random.normal(0, noise_scale, image.shape).astype('uint8')

def apply_adversarial_noise(image, noise, weight=0.1):
    """
    Apply weighted noise to create an adversarial image.
    
    Args:
        image: Input image
        noise: Noise array
        weight: Weight of the noise
    
    Returns:
        Adversarial image
    """
    return cv2.addWeighted(image, 1.0, noise, weight, 0)

def apply_diffusion_step(image, noise_level=0.1):
    """
    Apply one step of the forward diffusion process.
    
    Args:
        image: Input image
        noise_level: Amount of noise to add
    
    Returns:
        Image after diffusion step
    """
    # Normalize image to [0, 1]
    img_normalized = image.astype(float) / 255.0
    
    # Add noise
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(img_normalized + noise, 0, 1)
    
    # Convert back to uint8
    return (noisy_image * 255).astype('uint8')

def denoise_image(image, kernel_size=(5, 5)):
    """
    Apply denoising to an image.
    
    Args:
        image: Input noisy image
        kernel_size: Size of Gaussian kernel
    
    Returns:
        Denoised image
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def save_comparison_image(images, save_path):
    """
    Create and save a comparison image of all steps.
    
    Args:
        images: Dictionary of images to compare
        save_path: Path to save the comparison image
    """
    # Calculate grid dimensions
    n_images = len(images)
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    # Get maximum dimensions
    max_h = max(img.shape[0] for img in images.values())
    max_w = max(img.shape[1] for img in images.values())
    
    # Create comparison image
    comparison = np.zeros((max_h * grid_size, max_w * grid_size), dtype=np.uint8)
    
    # Place images in grid
    for idx, (name, img) in enumerate(images.items()):
        i, j = divmod(idx, grid_size)
        y1, y2 = i * max_h, (i + 1) * max_h
        x1, x2 = j * max_w, (j + 1) * max_w
        
        # Center the image in its grid cell
        y_offset = (max_h - img.shape[0]) // 2
        x_offset = (max_w - img.shape[1]) // 2
        comparison[y1 + y_offset:y1 + y_offset + img.shape[0],
                  x1 + x_offset:x1 + x_offset + img.shape[1]] = img
        
        # Add text label
        cv2.putText(comparison, name, (x1 + 10, y1 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    cv2.imwrite(save_path, comparison)
    return comparison