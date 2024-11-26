import glob
import json
import os

import cv2 as cv
import numpy as np

from .config import *


class SingleCameraCalibration:
    def __init__(self, left_or_right):
        self.left_or_right = left_or_right
        self.cam_output_dir = os.path.join(
            CALIBRATION_OUTPUT_DIR,
            (
                LEFT_CAMERA_CALIBRATION_FILENAME
                if left_or_right == "left"
                else RIGHT_CAMERA_CALIBRATION_FILENAME
            ),
        )
        self.chessboard_size = CHESSBOARD_SIZE
        self.frame_size = FRAME_SIZE
        self.chessboard_square_size_mm = CHESSBOARD_SQUARE_SIZE_MM
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.img_corners = []
        self.objpoints = []

    def set_objp(self):
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)
        self.objp = objp * self.chessboard_square_size_mm

    def preprocess_img(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return gray

    def find_corners(self, gray_img):
        ret, corners = cv.findChessboardCorners(gray_img, self.chessboard_size, None)
        if ret:
            corners = cv.cornerSubPix(
                gray_img, corners, (11, 11), (-1, -1), self.criteria
            )
            self.img_corners.append(corners)
            self.objpoints.append(self.objp)
        return ret, corners

    def visualize_corners(self, img, corners, ret):
        if ret:
            cv.drawChessboardCorners(img, self.chessboard_size, corners, ret)
            cv.imshow("img", img)
            cv.waitKey(1000)
        cv.destroyAllWindows()

    def calibrate(self):
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            self.objpoints, self.img_corners, self.frame_size, None, None
        )
        return ret, mtx, dist, rvecs, tvecs

    def save_calibration(self, mtx, dist):
        data = {"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist()}
        with open(self.cam_output_dir, "w") as f:
            json.dump(data, f, indent=4)


class StereoCameraCalibration:
    def __init__(self, mtx_left, dist_left, mtx_right, dist_right):
        self.mtx_left = mtx_left
        self.dist_left = dist_left
        self.mtx_right = mtx_right
        self.dist_right = dist_right
        self.output_dir = os.path.join(
            CALIBRATION_OUTPUT_DIR, STEREO_CAMERA_CALIBRATION_FILENAME
        )
        self.chessboard_size = CHESSBOARD_SIZE
        self.frame_size = FRAME_SIZE
        self.chessboard_square_size_mm = CHESSBOARD_SQUARE_SIZE_MM
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objpoints = []
        self.imgpoints_left = []
        self.imgpoints_right = []

    def set_objp(self):
        objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)
        self.objp = objp * self.chessboard_square_size_mm

    def preprocess_img(self, left_img, right_img):
        gray_left = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)
        return gray_left, gray_right

    def find_corners(self, gray_left, gray_right):
        ret_left, corners_left = cv.findChessboardCorners(
            gray_left, self.chessboard_size, None
        )
        ret_right, corners_right = cv.findChessboardCorners(
            gray_right, self.chessboard_size, None
        )
        if ret_left and ret_right:
            corners_left = cv.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), self.criteria
            )
            corners_right = cv.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), self.criteria
            )
            self.imgpoints_left.append(corners_left)
            self.imgpoints_right.append(corners_right)
            self.objpoints.append(self.objp)
        return ret_left, corners_left, ret_right, corners_right

    def visualize_corners(
        self, left_img, right_img, corners_left, corners_right, ret_left, ret_right
    ):
        if ret_left and ret_right:
            cv.drawChessboardCorners(
                left_img, self.chessboard_size, corners_left, ret_left
            )
            cv.drawChessboardCorners(
                right_img, self.chessboard_size, corners_right, ret_right
            )
            cv.imshow("Left img", left_img)
            cv.imshow("Right img", right_img)
            cv.waitKey(1000)
        cv.destroyAllWindows()

    def stereo_calibrate(self):
        flags = cv.CALIB_FIX_INTRINSIC
        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = (
            cv.stereoCalibrate(
                self.objpoints,
                self.imgpoints_left,
                self.imgpoints_right,
                self.mtx_left,
                self.dist_left,
                self.mtx_right,
                self.dist_right,
                self.frame_size,
                None,
                None,
                None,
                None,
                flags,
            )
        )
        return R, T, E, F

    def save_calibration(self, R, T, E, F):
        data = {"R": R.tolist(), "T": T.tolist(), "E": E.tolist(), "F": F.tolist()}
        with open(self.output_dir, "w") as f:
            json.dump(data, f, indent=4)


def singleCameraCalibration(left_or_right):
    img_dir = LEFT_CAMERA_IMG_DIR if left_or_right == "left" else RIGHT_CAMERA_IMG_DIR
    single_cam_cal = SingleCameraCalibration(left_or_right)

    for img_path in glob.glob(img_dir + "/*.png"):
        print(img_path)
        img = cv.imread(img_path)
        single_cam_cal.set_objp()
        gray_img = single_cam_cal.preprocess_img(img)
        ret, corners = single_cam_cal.find_corners(gray_img)
        single_cam_cal.visualize_corners(img, corners, ret)
        _, mtx, dist, _, _ = single_cam_cal.calibrate()
        single_cam_cal.save_calibration(mtx, dist)

    print(f"{left_or_right} camera calibration done")


def stereoCameraCalibration():
    left_cam_params = json.load(
        open(os.path.join(CALIBRATION_OUTPUT_DIR, LEFT_CAMERA_CALIBRATION_FILENAME))
    )
    right_cam_params = json.load(
        open(os.path.join(CALIBRATION_OUTPUT_DIR, RIGHT_CAMERA_CALIBRATION_FILENAME))
    )
    camera_matrix_left = np.array(left_cam_params["camera_matrix"])
    dist_coeffs_left = np.array(left_cam_params["dist_coeff"])
    camera_matrix_right = np.array(right_cam_params["camera_matrix"])
    dist_coeffs_right = np.array(right_cam_params["dist_coeff"])
    stereo_cam_cal = StereoCameraCalibration(
        camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right
    )

    images_left = sorted(
        glob.glob(os.path.join(STEREO_CALIBRATION_LEFT_IMG_DIR, "*.jpg"))
    )
    images_right = sorted(
        glob.glob(os.path.join(STEREO_CALIBRATION_RIGHT_IMG_DIR, "*.jpg"))
    )

    for img_left_path, img_right_path in zip(images_left, images_right):
        left_img = cv.imread(img_left_path)
        right_img = cv.imread(img_right_path)

        stereo_cam_cal.set_objp()
        gray_left, gray_right = stereo_cam_cal.preprocess_img(left_img, right_img)
        ret_left, corners_left, ret_right, corners_right = stereo_cam_cal.find_corners(
            gray_left, gray_right
        )
        stereo_cam_cal.visualize_corners(
            left_img, right_img, corners_left, corners_right, ret_left, ret_right
        )
        R, T, E, F = stereo_cam_cal.stereo_calibrate()
        stereo_cam_cal.save_calibration(R, T, E, F)

    print("Stereo camera calibration done")
