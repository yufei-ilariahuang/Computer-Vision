import cv2
import numpy as np
from .StereoCamera import StereoCamera


def get_roi_crop(left_img, right_img, bbox):
    """
    only interested in the bbox area in the left image,
    because of the rectification, the right image is cropped to the same height as the left image
    and object in the right must shift to the left relative to its position in the left image
    """
    # calculate the roi crop
    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    img_left_cropped = left_img[y_min:y_max, :x_max]
    img_right_cropped = right_img[y_min:y_max, :x_max]

    # initialize crop records
    left_crop_top, left_crop_bottom, left_crop_left, left_crop_right = (
        y_min,
        left_img.shape[0] - y_max,
        0,
        left_img.shape[1] - x_max,
    )
    right_crop_top, right_crop_bottom, right_crop_left, right_crop_right = (
        y_min,
        right_img.shape[0] - y_max,
        0,
        right_img.shape[1] - x_max,
    )

    # return cropped images and crop records
    return (
        img_left_cropped,
        img_right_cropped,
        (left_crop_top, left_crop_bottom, left_crop_left, left_crop_right),
        (right_crop_top, right_crop_bottom, right_crop_left, right_crop_right),
    )


def preprocess(left_img, right_img):
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # histogram equalization
    left_img = cv2.equalizeHist(left_img)
    right_img = cv2.equalizeHist(right_img)

    # bilateral filter
    left_img = cv2.bilateralFilter(left_img, 5, 75, 75)
    right_img = cv2.bilateralFilter(right_img, 5, 75, 75)

    # # edge enhancement
    # sobelx_left = cv2.Sobel(left_img, cv2.CV_64F, 1, 0, ksize=5)
    # sobely_left = cv2.Sobel(left_img, cv2.CV_64F, 0, 1, ksize=5)
    # left_img = cv2.magnitude(sobelx_left, sobely_left)
    # left_img = cv2.convertScaleAbs(left_img)

    # sobelx_right = cv2.Sobel(right_img, cv2.CV_64F, 1, 0, ksize=5)
    # sobely_right = cv2.Sobel(right_img, cv2.CV_64F, 0, 1, ksize=5)
    # right_img = cv2.magnitude(sobelx_right, sobely_right)
    # right_img = cv2.convertScaleAbs(right_img)

    return left_img, right_img


def undistortion(
    left_img,
    right_img,
    left_cam_matrix,
    left_dist_coeff,
    right_cam_matrix,
    right_dist_coeff,
):
    if (
        left_img is not None
        and left_cam_matrix is not None
        and left_dist_coeff is not None
    ):
        undistorted_left_img = cv2.undistort(left_img, left_cam_matrix, left_dist_coeff)
    else:
        undistorted_left_img = None

    if (
        right_img is not None
        and right_cam_matrix is not None
        and right_dist_coeff is not None
    ):
        undistorted_right_img = cv2.undistort(
            right_img, right_cam_matrix, right_dist_coeff
        )
    else:
        undistorted_right_img = None

    return undistorted_left_img, undistorted_right_img


def getRectifyTransform(height, width, stereoCamera: StereoCamera):
    # read intrinsic and extrinsic parameters
    left_K = stereoCamera.cam_matrix_left
    right_K = stereoCamera.cam_matrix_right
    left_distortion = stereoCamera.distortion_l
    right_distortion = stereoCamera.distortion_r
    R = stereoCamera.R
    T = stereoCamera.T

    # calculate rectification transform
    R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(
        left_K,
        left_distortion,
        right_K,
        right_distortion,
        (width, height),
        R,
        T,
        alpha=0,
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        left_K, left_distortion, R_left, P_left, (width, height), cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        right_K, right_distortion, R_right, P_right, (width, height), cv2.CV_32FC1
    )

    return map_left_x, map_left_y, map_right_x, map_right_y, Q


# distortion correction and stereo rectification
def rectifyImage(left_img, right_img, map_left_x, map_left_y, map_right_x, map_right_y):
    rectified_left_img = cv2.remap(left_img, map_left_x, map_left_y, cv2.INTER_AREA)
    rectified_right_img = cv2.remap(right_img, map_right_x, map_right_y, cv2.INTER_AREA)

    return rectified_left_img, rectified_right_img


def stereoMatchSGBM(left_img, right_img, bbox):
    # SGBM matching parameter settings
    img_channels = 3
    blockSize = 5
    paraml = {
        "minDisparity": 0,
        "numDisparities": 128,
        "blockSize": blockSize,
        "P1": 8 * img_channels * blockSize**2,
        "P2": 32 * img_channels * blockSize**2,
        "disp12MaxDiff": 5,
        "preFilterCap": 63,
        "uniquenessRatio": 15,
        "speckleWindowSize": 100,
        "speckleRange": 1,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    }

    # crop the image by bbox ROI
    left_img_cropped, right_img_cropped, left_crop_record, right_crop_record = (
        get_roi_crop(left_img, right_img, bbox)
    )
    cropped_size = (left_img_cropped.shape[1], left_img_cropped.shape[0])
    numDisparities = min((cropped_size[0] // 8) * 8, paraml["numDisparities"])
    paraml["numDisparities"] = numDisparities

    # build SGBM object
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr["minDisparity"] = -paraml[
        "numDisparities"
    ]  # set search direction to the opposite of the left image
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # downscale the image
    left_img_down = cv2.pyrDown(left_img_cropped)
    right_img_down = cv2.pyrDown(right_img_cropped)
    downscale_factor = cropped_size[0] / left_img_down.shape[1]

    try:
        disparity_left_half = left_matcher.compute(left_img_down, right_img_down)
        disparity_right_half = right_matcher.compute(right_img_down, left_img_down)
    except Exception as e:
        print(f"stereo matching error: {e}")
        return None, None

    # resize the disparity map to the original size
    disparity_left = cv2.resize(
        disparity_left_half, cropped_size, interpolation=cv2.INTER_AREA
    )
    disparity_right = cv2.resize(
        disparity_right_half, cropped_size, interpolation=cv2.INTER_AREA
    )

    # scale the disparity map to the original size
    disparity_left = downscale_factor * disparity_left
    disparity_right = downscale_factor * disparity_right

    # restore the cropped area
    disparity_left = cv2.copyMakeBorder(
        disparity_left,
        left_crop_record[0],
        left_crop_record[1],
        left_crop_record[2],
        left_crop_record[3],
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    disparity_right = cv2.copyMakeBorder(
        disparity_right,
        right_crop_record[0],
        right_crop_record[1],
        right_crop_record[2],
        right_crop_record[3],
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    # scale the disparity map to the original size, because the disparity obtained by the SGBM algorithm is ×16
    disparity_left = disparity_left.astype(np.float32) / 16.0
    disparity_right = disparity_right.astype(np.float32) / 16.0

    return disparity_left, disparity_right
