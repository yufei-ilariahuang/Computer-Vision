from time import time

import cv2
import numpy as np

from .utils.stereo_vision_processing import (
    getRectifyTransform,
    preprocess,
    rectifyImage,
    stereoMatchSGBM,
    undistortion,
)
from .utils.StereoCamera import StereoCamera


def get_measure_points(bbox, mode="single"):
    """
    get the center point of the bbox
    """
    center_point = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
    if mode == "single":
        return [center_point]
    elif mode == "average":
        # according to the size of the bounding box to determine the number of sampling points
        num_points_x = min(4, int((bbox[2] - bbox[0]) // 20))
        num_points_y = min(4, int((bbox[3] - bbox[1]) // 20))

        points = []
        for i in range(num_points_x):
            for j in range(num_points_y):
                x = bbox[0] + (i + 0.5) * (bbox[2] - bbox[0]) / num_points_x
                y = bbox[1] + (j + 0.5) * (bbox[3] - bbox[1]) / num_points_y
                points.append((int(x), int(y)))
        points.append(center_point)
        return points


def calc_distance(img_left, img_right, bbox):
    """
    calculate the distance of a certain point in the image

    img_left: original left image after undistortion
    img_right: original right image after undistortion
    bbox: the bbox to measure distance, (x_min, y_min, x_max, y_max)
    """
    height, width = img_left.shape[0:2]

    # preprocess, undistort and rectify image
    config = StereoCamera()
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    img_left_preprocessed, img_right_preprocessed = preprocess(img_left, img_right)
    img_left_undistorted, img_right_undistorted = undistortion(
        img_left_preprocessed,
        img_right_preprocessed,
        config.cam_matrix_left,
        config.distortion_l,
        config.cam_matrix_right,
        config.distortion_r,
    )
    img_left_rectified, img_right_rectified = rectifyImage(
        img_left_undistorted, img_right_undistorted, map1x, map1y, map2x, map2y
    )

    # crop image and get measure points
    measure_points = get_measure_points(bbox, mode="average")

    # stereo matching
    disp, _ = stereoMatchSGBM(img_left_rectified, img_right_rectified, bbox)
    if disp is None:
        return None
    points_3d = cv2.reprojectImageTo3D(disp, Q)
    distances_np = np.zeros(len(measure_points))

    for i, measure_point in enumerate(measure_points):
        dis = (
            (
                points_3d[measure_point[1], measure_point[0], 0] ** 2
                + points_3d[measure_point[1], measure_point[0], 1] ** 2
                + points_3d[measure_point[1], measure_point[0], 2] ** 2
            )
            ** 0.5
        ) / 1000  # in meters
        distances_np[i] = dis

    filtered_distance = filter_distance(distances_np)

    return filtered_distance


def filter_distance(distances, distance_threshold=30):
    """
    filter the distance

    distances: the distances numpy array to filter
    distance_threshold: the maximum distance to filter
    """
    filtered_distances = distances[distances < distance_threshold]

    # if no valid distance values, return None
    if len(filtered_distances) == 0:
        return None

    # calculate the quartiles to remove outliers
    q1 = np.percentile(filtered_distances, 25)
    q3 = np.percentile(filtered_distances, 75)
    iqr = q3 - q1

    # define the bounds of outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # remove outliers
    filtered_distances = filtered_distances[
        (filtered_distances >= lower_bound) & (filtered_distances <= upper_bound)
    ]

    # if no valid distance values after removing outliers, return None
    if len(filtered_distances) == 0:
        return None

    # return the mean
    return np.mean(filtered_distances)
