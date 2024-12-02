import cv2
import numpy as np

def initialize_camera(camera_index=0):
    """
    Initialize the camera capture.
    
    Args:
        camera_index: Index of camera (0 for default, 1 for external)
    
    Returns:
        VideoCapture object
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    # Add a small delay to allow camera to initialize
    cv2.waitKey(1000)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video dimensions: {width}x{height}")
    return cap

def get_optical_flow_params():
    """
    Get parameters for optical flow calculations.
    
    Returns:
        Dictionary of parameters for feature detection and optical flow
    """
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.5,    # Increased for stronger corners
        minDistance=15,      # Increased to spread points
        blockSize=11         # Increased for better detection
    )
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(21, 21),    # Adjusted window size
        maxLevel=3,          # Increased pyramid levels
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    
    return feature_params, lk_params

def filter_excessive_motion(good_new, good_old, max_displacement=30):
    """Filter out points with excessive motion."""
    if len(good_new) == 0 or len(good_old) == 0:
        return good_new, good_old
    distances = np.sqrt(np.sum((good_new - good_old) ** 2, axis=1))
    valid_mask = distances < max_displacement
    return good_new[valid_mask], good_old[valid_mask]


def process_sparse_optical_flow(old_gray, frame_gray, p0, lk_params):
    """
    Calculate sparse optical flow using Lucas-Kanade method.
    
    Args:
        old_gray: Previous frame in grayscale
        frame_gray: Current frame in grayscale
        p0: Previous points to track
        lk_params: Parameters for Lucas-Kanade optical flow
        
    Returns:
        Tuple of (new points, status, error)
    """
    return cv2.calcOpticalFlowPyrLK(
        old_gray, 
        frame_gray, 
        p0, 
        None, 
        **lk_params
    )

def process_dense_optical_flow(old_gray, frame_gray):
    """
    Calculate dense optical flow using Farneback method.
    
    Args:
        old_gray: Previous frame in grayscale
        frame_gray: Current frame in grayscale
        
    Returns:
        Flow vectors
    """
    return cv2.calcOpticalFlowFarneback(
        old_gray,
        frame_gray,
        None,
        0.5,  # pyr_scale
        4,    # levels
        15,   # winsize
        3,    # iterations
        5,    # poly_n
        1.2,  # poly_sigma
        0     # flags
    )

def draw_flow_tracks(frame, mask, good_new, good_old):
    """
    Draw optical flow tracks on the frame.
    
    Args:
        frame: Current frame
        mask: Mask for drawing tracks
        good_new: New tracked points
        good_old: Previous tracked points
        
    Returns:
        Frame with tracks drawn
    """
    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # Thicker lines for better visibility
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        # Larger circles for better visibility
        frame = cv2.circle(frame, (int(a), int(b)), 3, (0, 0, 255), -1)
    
    output = cv2.add(frame, mask)
    return output, mask

def visualize_dense_flow(flow):
    """
    Convert dense optical flow to visualization.
    
    Args:
        flow: Dense optical flow vectors
        
    Returns:
        Visualization of flow in BGR format
    """
    # Convert flow to polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV visualization
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue represents direction
    hsv[..., 1] = 255                     # Full saturation
    # Improved normalization for better visualization
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to BGR
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)