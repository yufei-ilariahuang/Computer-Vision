# Real-Time Feature Matching with Dual Webcams

This project demonstrates real-time feature detection and matching across different camera views using Python and OpenCV. It features dynamic keyboard controls that allow users to toggle between SIFT and ORB feature detection methods and switch between Brute-Force and FLANN-based matching algorithms. Key presses instantly adjust the algorithms, enabling direct comparison and visualization of different methods’ performance on-screen. 


## Team Members
- Siyi Gao
- Yufei Huang
- Xuedong Pan
- Tongle Yao

## Video Explanation
- https://northeastern.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=2715ec05-98e5-4c86-a911-b1fa01801a38


## Project Requirements
- Python 3.x
- OpenCV library (cv2)
- Numpy

## Setup Instructions
1. Clone the repository:
   ```
   git clone https://github.com/siyigao/mini-project4.git
   cd mini-project4
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python WebCamDual.py
   ```

## Usage
- The application will display two webcam feeds side by side.
- default feature detection method: SIFT
- default matching method: Brute-Force Matching
- 's' Key: Press 's' to switch the feature detection method to SIFT, ideal for precision in stable conditions.
- 'o' Key: Press 'o' to switch to ORB for faster feature detection, suitable for real-time applications.
- 'b' Key: Press 'b' to activate Brute-Force Matching, providing exhaustive and accurate matches.
- 'f' Key: Press 'f' to employ FLANN-based Matching, enhancing speed and efficiency in large datasets.
- 'q' Key: Press 'q' to exit the application.

## Conclusion
- ORB is the fastest algorithm while SIFT performs the best in the most scenarios. For special case when the angle of rotation is proportional to 90 degrees, ORB outperforms SIFT, and in the noisy images, ORB and SIFT show almost similar performances. In ORB, the features are mostly concentrated in objects at the center of the image,while in SIFT key point detectors are distributed over the image.
- FLANN-based matching is faster than Brute-Force Matching.
