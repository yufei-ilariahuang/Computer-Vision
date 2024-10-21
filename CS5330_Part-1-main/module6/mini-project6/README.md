# Real-Time Panorama Creation Using Live Camera
The goal of this mini-project is to develop a lane detection system for identifying lane lines on a road
using the Hough Line Transform. The system will be implemented using a given starter code
(WebCamSave.py) and will include necessary preprocessing and post-processing steps to accurately
detect the two-lane lines (left and right) for driving a car. The detected lanes will be highlighted with red lines in the output. 

## Team Members
- Siyi Gao
- Yufei Huang
- Xuedong Pan
- Tongle Yao

# Code Demo Video Link
https://youtu.be/jsZZ4PGZK1k

# Road Test Video Link
https://youtu.be/tKbgnrYfakc

## Setup Instructions
1. Clone the repository:
   ```
   git clone https://github.khoury.northeastern.edu/panxuedong418/CS5330_F24_Group5.git
   cd mini-project5
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install numpy opencv-python
   ```

4. Run the application:
   ```
   python WebCamSave.py  # use the default camera
   ```
   OR
   ```
   python WebCamSave.py -f <file_path>  # use the specified video file
   ```
   OR
   ```
   python WebCamSave.py -f <file_path> -o <output_file_path>  # use the specified video file and output file path
   ```
   OR
   ```
   python WebCamSave.py -o <output_file_path>  # use the default camera and the specified output file path
   ```

## Usage
- Press 'p' to pause/resume the video.
- Press 'a' to show Gaussian Blur.
- Press 'w' to show Filter White Lines.
- Press 'm' to show Masked by ROI.
- Press 'e' to show Edges detected by Canny Edge Detection.
- Press 'l' to show Lines detected by Hough Line Transform.
- Press 'o' to show final output frame.
- Press '0' to show Original frame.
- Press 'q' to quit the application.
