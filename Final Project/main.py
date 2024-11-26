import argparse
import multiprocessing
import os
import queue
import signal
import sys
from time import sleep

import cv2

from camera.calculate_distance import calc_distance, get_measure_points
from frame_processing.plot_frame import plot_bbox, plot_class_confidence, plot_distance, plot_fps
from frame_processing.post_processing import convert_bbox_to_origin_img
from frame_processing.pre_processing import preprocess_image
from training.predict import predict
from training.utils import get_device, load_label_map, load_model
from tts.tts import tts_closest_object, tts_process_function

SKIP_FRAME_COUNT = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--label", type=str, required=True, help="label map path")
    parser.add_argument(
        "--left_cam", type=int, required=True, help="index of the left camera"
    )
    parser.add_argument(
        "--right_cam", type=int, required=True, help="index of the right camera"
    )
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="iou threshold")
    parser.add_argument("--half", type=bool, default=False, help="use half precision")
    parser.add_argument(
        "--augment", type=bool, default=False, help="enable augmentation"
    )
    args = parser.parse_args()

    model_path = args.model
    label_path = args.label
    cam_idx_l = args.left_cam
    cam_idx_r = args.right_cam
    conf = args.conf
    iou = args.iou
    half = args.half
    augment = args.augment

    return model_path, label_path, cam_idx_l, cam_idx_r, conf, iou, half, augment


def capture_frames(cap_l, cap_r):
    ret_l, img_l = cap_l.read()
    ret_r, img_r = cap_r.read()

    if not ret_l or not ret_r:
        return None, None
    return img_l, img_r


def waiting_reconnect(cam_idx_l, cam_idx_r):
    print("camera is not connected, waiting reconnect...")
    for remaining in range(30, 0, -1):
        cap_l = cv2.VideoCapture(cam_idx_l)
        cap_r = cv2.VideoCapture(cam_idx_r)
        if cap_l.isOpened() and cap_r.isOpened():
            img_l, img_r = capture_frames(cap_l, cap_r)
            if img_l is not None and img_r is not None:
                print("success to reconnect camera")
                return True
        sys.stdout.write(f"\rwaiting reconnect... ({remaining}s)")
        sys.stdout.flush()
        sleep(1)
    return False


def signal_handler(sig, frame, message_queue, tts_process):
    print("program is terminating...")
    message_queue.put(None)
    tts_process.join()
    sys.exit(0)


if __name__ == "__main__":
    # set the number of threads for opencv
    num_cores = os.cpu_count()
    cv2.setNumThreads(num_cores - 1)

    # initialization
    model_path, label_path, cam_idx_l, cam_idx_r, conf, iou, half, augment = (
        parse_args()
    )
    device = get_device()
    model, model_type = load_model(model_path)
    label_map = load_label_map(label_path, model_type)
    cap_l = cv2.VideoCapture(cam_idx_l)
    cap_r = cv2.VideoCapture(cam_idx_r)

    # create a queue to store the information to be reported
    message_queue = queue.Queue()

    # 创建一个多进程队列
    message_queue = multiprocessing.Queue()

    # 启动 TTS 进程
    tts_process = multiprocessing.Process(
        target=tts_process_function, args=(message_queue,)
    )
    tts_process.start()

    # register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # save for skip frame
    origin_bboxes = []
    predict_classes = []
    predict_confidences = []
    distances = []
    skip_frame_count = SKIP_FRAME_COUNT

    # main loop
    while True:
        # capture frames
        frame_l, frame_r = capture_frames(cap_l, cap_r)

        if frame_l is None or frame_r is None:
            message_queue.put(None)
            print("camera is not connected, exit...")
            exit(1)
    
        preprocess_img_l = preprocess_image(frame_l)
        preprocess_img_r = preprocess_image(frame_r)

        if skip_frame_count == 0:
            # predict
            pred = predict(
                preprocess_img_l, model, conf, iou, half, augment, model_type
            )
            predict_bboxes = pred.boxes.xyxy.cpu().numpy()
            predict_classes = pred.boxes.cls.cpu().numpy()
            predict_confidences = pred.boxes.conf.cpu().numpy()
            origin_bboxes = convert_bbox_to_origin_img(
                predict_bboxes, frame_l.shape, (640, 640)
            )
            distances = []

        # calculate distance and plot
        for i, (origin_bbox, predict_class, predict_confidence) in enumerate(
            zip(origin_bboxes, predict_classes, predict_confidences)
        ):
            # if predict_class == 0:
            #     continue

            # calculate distance
            if skip_frame_count == 0:
                distance = calc_distance(frame_l, frame_r, origin_bbox)
                distances.append((distance, label_map[int(predict_class)]))
            else:
                distance = distances[i][0] if len(distances) > i else None

            # plot bbox, class, distance
            frame_l = plot_bbox(frame_l, origin_bbox)
            frame_l = plot_class_confidence(
                frame_l,
                label_map[int(predict_class)],
                predict_confidence,
                origin_bbox,
                (255, 0, 0),
            )
            frame_l = plot_distance(frame_l, distance, origin_bbox, (255, 0, 0))

        if skip_frame_count == 0:
            skip_frame_count = SKIP_FRAME_COUNT
        else:
            skip_frame_count -= 1

        cv2.imshow("img", frame_l)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("v"):
            print("tts start...")
            tts_closest_object(distances, message_queue)
            print("tts end...")

    # put None to the queue to stop the TTS process
    message_queue.put(None)
    tts_process.join()
