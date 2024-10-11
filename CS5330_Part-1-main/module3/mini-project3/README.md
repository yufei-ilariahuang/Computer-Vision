# Mini-project 3: Image World App
## Team Members
- Siyi Gao
- Yufei Huang
- Xuedong Pan
- Tongle Yao

## Project Requirements
- Python 3.x
- OpenCV library (cv2)
- Numpy

## Launching the Application
`python3 WebCam2.py`

When launching the program, it will call launch the default ID=0 camera

When the application starts, you will see a window displaying the webcam feed on left and right sides. Left side will display original camera feed, and right side will display the warped camera feed. The FPS is displayed on the left-top corner, and the current active warp mode is displayed on the left-top corner of the right side camera feed.

## Keyboard Guide
- press "q" to quite the application
- press "t" to toggle translation mode
- Press "r" to toggle rotation mode
- Press "s" to toggle scaling mode
- Press "p" to enable perspective transformation modes
- "y", "h", "g", and "j" are mapped to UP, DOWN, LEFT, RIGHT buttons
- In Translation Mode:
    - UP (y): move up by 1 pixel
    - DOWN (h): move down by 1 pixel
    - LEFT (g): move left by 1 pixel
    - RIGHT (j): move right by 1 pixel
- In Rotation Mode:
    - LEFT (g): rotate counter-clockwise by 1 degree
    - RIGHT (j): rotate clockwise by 1 degree
- In Scaling Mode:
    - UP (y): Scale up by 1%
    - Down (h): Scale down by 1%
- In Perspective Transformation Mode:
    - Fixed angle of perspective transformation, no interations

Demo video1 link: https://drive.google.com/file/d/1GlRtDHhILOVux8bu6Z18zaK5Pt-yPx4W/view?usp=sharing

Demo video2 link (demo with plug-in webcam): https://drive.google.com/file/d/1kFPKsn2lns9mf-aVL8KnD0pSZOl_oV3b/view?usp=sharing
