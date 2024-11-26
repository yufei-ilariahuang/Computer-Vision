import json
import os

import numpy as np

from .config import *


class StereoCamera(object):
    def __init__(self):
        left_cam_params = json.load(
            open(os.path.join(CALIBRATION_OUTPUT_DIR, LEFT_CAMERA_CALIBRATION_FILENAME))
        )
        right_cam_params = json.load(
            open(
                os.path.join(CALIBRATION_OUTPUT_DIR, RIGHT_CAMERA_CALIBRATION_FILENAME)
            )
        )
        stereo_cam_params = json.load(
            open(
                os.path.join(CALIBRATION_OUTPUT_DIR, STEREO_CAMERA_CALIBRATION_FILENAME)
            )
        )

        # left camera intrinsic matrix
        self.cam_matrix_left = np.array(left_cam_params["camera_matrix"])
        # right camera intrinsic matrix
        self.cam_matrix_right = np.array(right_cam_params["camera_matrix"])

        # distortion coefficients of left and right cameras: [k1, k2, p1, p2, k3]
        self.distortion_l = np.array(left_cam_params["dist_coeff"])
        self.distortion_r = np.array(right_cam_params["dist_coeff"])

        # rotation matrix
        self.R = np.array(stereo_cam_params["R"])

        # translation matrix
        self.T = np.array(stereo_cam_params["T"])

        # focal length
        self.focal_length = self.cam_matrix_left[
            0, 0
        ]  # default value, usually taken from the re-projection matrix Q

        # baseline distance
        self.baseline = self.T[
            0
        ]  # unit: mm, the first parameter of the translation vector (take the absolute value)
