import argparse
from camera.take_calibrate_photo import take_calibrate_photo_single, take_calibrate_photo_dual

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration")
    parser.add_argument("--mode", type=str, default="dual", required=True, help="calibrate mode, left, right or dual")
    parser.add_argument("--left_cam", type=int, default=0, required=True, help="left camera index")
    parser.add_argument("--right_cam", type=int, default=1, required=False, help="right camera index")
    parser.add_argument("--auto_take_photo", type=bool, default=False, required=False, help="auto take photo every 3 seconds or manually press 's'")
    args = parser.parse_args()
    
    mode = args.mode
    left_cam = args.left_cam
    right_cam = args.right_cam
    auto_take_photo = args.auto_take_photo

    if mode == "dual" and (left_cam is None or right_cam is None):
        print("Error: left_cam and right_cam are required for dual mode")
        exit(1)
    
    if mode == "left" or mode == "right":
        cam = left_cam if mode == "left" else right_cam
        take_calibrate_photo_single(cam, mode, auto_take_photo)
    elif mode == "dual":  # dual mode
        take_calibrate_photo_dual(left_cam, right_cam, auto_take_photo)
    else:
        print("Error: invalid mode")
        exit(1)
