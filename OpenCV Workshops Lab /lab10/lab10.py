import cv2
import numpy as np
from .camera_operations import (
    initialize_camera,
    get_optical_flow_params,
    process_sparse_optical_flow,
    process_dense_optical_flow,
    draw_flow_tracks,
    visualize_dense_flow,
    filter_excessive_motion
    
)

def lab10():
    try:
        print("Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
            
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.2,
            minDistance=10,
            blockSize=15
        )
        
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        tracking_active = False
        frames_since_reset = 0
        reset_interval = 30

        import os
        save_dir = 'lab10'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        ret, first_frame = cap.read()
        mask = np.zeros_like(first_frame)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if tracking_active:
                frames_since_reset += 1
                if frames_since_reset >= reset_interval or 'p0' not in locals() or p0 is None or len(p0) < 10:
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                    frames_since_reset = 0
                    mask = np.zeros_like(frame)
                
                if p0 is not None and len(p0) > 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                    
                    if p1 is not None:
                        good_new = p1[st == 1]
                        good_old = p0[st == 1]
                        
                        # Remove motion filtering to see all tracks
                        if len(good_new) > 0 and len(good_old) > 0:
                            # Draw the tracks
                            img = frame.copy()
                            
                            # Draw tracks with brighter colors and thicker lines
                            for i, (new, old) in enumerate(zip(good_new, good_old)):
                                a, b = new.ravel()
                                c, d = old.ravel()
                                
                                # Brighter green color for lines
                                mask = cv2.line(mask, (int(c), int(d)), (int(a), int(b)),
                                              (0, 255, 0), 2)
                                
                                # Larger, brighter red circles
                                img = cv2.circle(img, (int(a), int(b)), 5, (0, 0, 255), -1)
                            
                            # Combine with less fade
                            sparse_flow_img = cv2.addWeighted(img, 0.7, mask, 0.9, 0)
                            
                            # Calculate dense flow
                            dense_flow = cv2.calcOpticalFlowFarneback(
                                old_gray, frame_gray, None, 
                                0.5, 3, 15, 3, 5, 1.2, 0)
                            
                            mag, ang = cv2.cartToPolar(dense_flow[..., 0], dense_flow[..., 1])
                            hsv = np.zeros_like(frame)
                            hsv[..., 0] = ang * 180 / np.pi / 2
                            hsv[..., 1] = 255
                            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                            dense_flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                            
                            cv2.imshow('Original Frame', frame)
                            cv2.imshow('Sparse Optical Flow', sparse_flow_img)
                            cv2.imshow('Dense Optical Flow', dense_flow_img)
                            
                            # Slower fade for the mask
                            mask = cv2.addWeighted(mask, 0.9, np.zeros_like(mask), 0.1, 0)
                            
                            old_gray = frame_gray.copy()
                            p0 = good_new.reshape(-1, 1, 2)
            
            else:
                cv2.imshow('Original Frame', frame)
                old_gray = frame_gray.copy()
            
            key = cv2.waitKey(1)
            if key == ord('s'):
                tracking_active = True
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                mask = np.zeros_like(frame)
                frames_since_reset = 0
                print("Tracking started")
                
            elif key == ord('q'):
                if tracking_active:
                    cv2.imwrite(os.path.join(save_dir, 'original.jpg'), frame)
                    cv2.imwrite(os.path.join(save_dir, 'sparse_flow.jpg'), sparse_flow_img)
                    cv2.imwrite(os.path.join(save_dir, 'dense_flow.jpg'), dense_flow_img)
                    
                    comparison = np.zeros((max(frame.shape[0]*2, frame.shape[0]), 
                                        frame.shape[1]*2, 3), dtype=np.uint8)
                    
                    comparison[:frame.shape[0], :frame.shape[1]] = frame
                    comparison[:frame.shape[0], frame.shape[1]:] = sparse_flow_img
                    comparison[frame.shape[0]:, :frame.shape[1]] = dense_flow_img
                    
                    cv2.imwrite(os.path.join(save_dir, 'comparison.jpg'), comparison)
                    print(f"Images saved in {save_dir}/")
                break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    lab10()