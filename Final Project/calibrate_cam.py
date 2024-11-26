import argparse
from camera.utils.calibration import singleCameraCalibration, stereoCameraCalibration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calibration")
    parser.add_argument("--mode", type=str, default="dual", required=True, help="calibrate mode, left, right or dual")
    args = parser.parse_args()
    
    calibrate_mode = args.mode

    if calibrate_mode == "left" or calibrate_mode == "right":
        singleCameraCalibration(calibrate_mode)
    elif calibrate_mode == "dual":
        stereoCameraCalibration()
    else:
        print("Invalid calibrate mode, please enter the correct mode")
