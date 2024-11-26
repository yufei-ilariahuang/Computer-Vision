import os
import time

import cv2
from .utils.config import *


def take_calibrate_photo_dual(left_cam, right_cam, auto_mode=False):
    """
    take calibrate photo from camera, you can hold a chessboard to calibrate the camera
    photo will be saved in images/calibrate_photo/
    """
    left_cap = cv2.VideoCapture(left_cam)
    right_cap = cv2.VideoCapture(right_cam)

    os.makedirs(STEREO_CALIBRATION_LEFT_IMG_DIR, exist_ok=True)
    os.makedirs(STEREO_CALIBRATION_RIGHT_IMG_DIR, exist_ok=True)

    last_capture_time = time.time()
    num = 0

    while left_cap.isOpened() and right_cap.isOpened():

        _, left_cam_frame = left_cap.read()
        _, right_cam_frame = right_cap.read()

        if auto_mode:
            current_time = time.time()
            if current_time - last_capture_time >= 3:
                cv2.imwrite(
                    os.path.join(STEREO_CALIBRATION_LEFT_IMG_DIR, str(num) + ".png"),
                    left_cam_frame,
                )
                cv2.imwrite(
                    os.path.join(STEREO_CALIBRATION_RIGHT_IMG_DIR, str(num) + ".png"),
                    right_cam_frame,
                )
                print("image saved!")
                num += 1
                last_capture_time = current_time

        k = cv2.waitKey(5) & 0xFF

        if k == ord("q"):
            break
        elif k == ord("s") and not auto_mode:  # wait for 's' key to save photo
            cv2.imwrite(
                os.path.join(STEREO_CALIBRATION_LEFT_IMG_DIR, str(num) + ".png"),
                left_cam_frame,
            )
            cv2.imwrite(
                os.path.join(STEREO_CALIBRATION_RIGHT_IMG_DIR, str(num) + ".png"),
                right_cam_frame,
            )
            print("image saved!")
            num += 1

        cv2.imshow("left_cam", left_cam_frame)
        cv2.imshow("right_cam", right_cam_frame)

    # Release and destroy all windows before termination
    left_cap.release()
    right_cap.release()


def take_calibrate_photo_single(cam, left_or_right, auto_mode=False):
    cap = cv2.VideoCapture(cam)
    last_capture_time = time.time()
    output_dir = (
        LEFT_CAMERA_IMG_DIR if left_or_right == "left" else RIGHT_CAMERA_IMG_DIR
    )
    os.makedirs(output_dir, exist_ok=True)
    num = 0

    while cap.isOpened():
        _, frame = cap.read()

        if auto_mode:
            current_time = time.time()
            if current_time - last_capture_time >= 3:
                cv2.imwrite(os.path.join(output_dir, str(num) + '.png'), frame)
                print("image saved!")
                num += 1
                last_capture_time = current_time
    
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s") and not auto_mode:
            cv2.imwrite(os.path.join(output_dir, str(num) + ".png"), frame)
            print("image saved!")
            num += 1

        cv2.imshow(f"{left_or_right} camera", frame)

    cap.release()
    cv2.destroyAllWindows()
