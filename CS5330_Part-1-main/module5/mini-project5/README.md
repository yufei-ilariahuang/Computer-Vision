# Real-Time Panorama Creation Using Live Camera
This mini-project involves developing a real-time application using Python and OpenCV that captures a sequence of images from a live camera to create a panorama. The user will initiate the panorama capture by pressing the ‘s’ key and moving the camera horizontally (e.g., from left to right). Once the desired area has been covered, pressing the ‘a’ key will stop the capture and trigger the generation of a panoramic image using a stitching algorithm.

## Team Members
- Siyi Gao
- Yufei Huang
- Xuedong Pan
- Tongle Yao

## Video Explanation
- https://drive.google.com/file/d/1RQb4btFpyZJMC3a4SnmsIOoRLKBTAETe/view?usp=sharing

## Another Demo: Generate Panorama using test.mov
- https://drive.google.com/file/d/1ERScbxmasF4a_O4__5MB2DdbFq77qbuQ/view?usp=sharing

## Project Requirements
- Python 3.x
- OpenCV library (cv2)
- Numpy

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
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python WebCam.py
   ```
   OR
   ```
   python WebCam.py -f <file_path>
   ```
   OR
   ```
   python WebCam.py --file <file_path>
   ```

## Usage
- The application will display a camera feed OR input video file with real-time FPS and current stitching algorithm.
- 's' Key: Start capturing frames for stitching, it will capture one frame every second.
- 'a' Key: Stop capturing frames. And the program will process the captured frames and stitch them together. Finally, a new window will be popped up with a preview of the panorama.
- 'd' Key: Close the preview of the panorama.
- 'f' Key: When you are in the preview window, it will save the previewed panorama as "panorama.jpg".
- 'm' Key: Switch stitching algorithm between SIFT and ORB. SIFT by default. 
- 'q' Key: Exit the whole application.

## Important Note
- When the input is a video file, the program will exit right after the video ends. Hence, be prepared to click 's' and 'a' to start and stop capturing frames for the panorama. Also, be prepared to click 'f' to save the panorama to the "panorama.jpg" file when the preview window pops up, if you want to save it.
- Move slowly and use SIFT mode will have a better panorama result.
- If the program exited accidently, please take a look at the error message in terminal.
- The "panorama.jpg" in the folder is the panorama generated using our program in Demo2.

## Conclusion
- Since we implemented both ORB and SIFT, we found SIFT has a better stitching result than ORB but it takes a longer processing time.
- As more images are stitched together, the left picture will become blurred, which will reduce the matches and increase mismatches between left and right images. Hence, stitching too many frames together will result in a terrible panorama picture.
- High-resolution video input will lead to a better panorama result, as we tested using the raw video (uncompressed) captured by phones, the resulting panorama looks much better.
- If the camera is moved horizontally instead of rotating around somewhere, the resulting panorama will be better since the camera perspective does not change a lot.
