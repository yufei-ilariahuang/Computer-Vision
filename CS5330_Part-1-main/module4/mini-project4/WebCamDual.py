import cv2
import time

class FPSCalculator:
    def __init__(self, update_interval=1):
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.update_interval = update_interval

    def update(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time

        if elapsed_time > self.update_interval:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        return self.fps

def feature_detection(frame, method='SIFT'):
    # Feature Detection and Description
    # Choose and implement either SIFT or ORB for feature detection and description
    # Detect features in the frames captured from both cameras.

    # SIFT

    if method == 'SIFT':
        detector = cv2.SIFT_create()
    # ORB
    # compare to SIFT, ORB is faster but less accurate, increasing nfeatures can make it a bit more accurate
    elif method == 'ORB':
        detector = cv2.ORB_create(nfeatures=2000)

    
    keypoint, descriptor = detector.detectAndCompute(frame, None)

    return keypoint, descriptor

def feature_matching(descriptor1, descriptor2, method='BF'):
    # Feature Matching
    # Match the features detected in the frames from both cameras.

    # Ensure descriptors are of the same type
    if descriptor1 is None or descriptor2 is None:
        return []

    if descriptor1.dtype != descriptor2.dtype:
        descriptor1 = descriptor1.astype(np.float32)
        descriptor2 = descriptor2.astype(np.float32)

    # Brute-Force Matcher
    if method == 'BF':
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    # FLANN
    elif method == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
    except cv2.error:
        # If knnMatch fails, fall back to simple matching
        matches = matcher.match(descriptor1, descriptor2)
        # Convert to a list of lists to maintain consistent structure
        matches = [[m] for m in matches]

    return matches


def find_best_match(matches, ratio):
    # Find the best match
    # n is the best match, m is the second best match
    # if the best match distance is less than 'ratio' times the second best match distance, then it is a good match
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def draw_matches(img1, img2, keypoint1, keypoint2, good_matches):
    # Draw lines connecting matched feature points between the frames from the two cameras.
    # Display the matched points with clear visualization, indicating their correspondence.
    img_matches = cv2.drawMatches(img1, keypoint1, img2, keypoint2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

def display_match_score(img_matches, good_matches, keypoint1):
    # for each good match, calculate the matching score, and display it on the line
    for match in good_matches:
        score = match.distance
        pt1 = keypoint1[match.queryIdx].pt
        x = int(pt1[0])
        y = int(pt1[1])
        cv2.putText(img_matches, f'Score: {int(score)}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img_matches

def process_frame(frame):
    # lower the resolution of the frame, can help increase the fps
    return cv2.resize(frame, (640, 480))

def main():
    # Initialize
    print("Initializing cameras...")
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    time.sleep(2)
    if not cap1.isOpened() or not cap2.isOpened():
        print("Failed to open one or both cameras.")
        return
    fps_calc = FPSCalculator()
    print("Cameras initialized. Entering main loop...")
    # Default method
    feature_method = 'SIFT'  
    matching_method = 'BF'
    while True:
        # Capture frame-by-frame from the first camera
        ret1, frame1 = cap1.read()
        # Capture frame-by-frame from the second camera
        ret2, frame2 = cap2.read()

        # Check if frames are captured
        if not ret1 or not ret2:
            break

        # process frame
        frame1 = process_frame(frame1)
        frame2 = process_frame(frame2)

        # Feature Detection and Description
        keypoint1, descriptor1 = feature_detection(frame1, feature_method)
        keypoint2, descriptor2 = feature_detection(frame2, feature_method)

        # Feature Matching
        matches = feature_matching(descriptor1, descriptor2, matching_method)

        # Find the best match
        ratio = 0.5
        #the closest match must be at least (1-ratio) times closer than the second closest
        good_matches = find_best_match(matches, ratio)

        # Visualization of Matched Features
        img_matches = draw_matches(frame1, frame2, keypoint1, keypoint2, good_matches)

        # display the matching score
        img_matches = display_match_score(img_matches, good_matches, keypoint1)

        # calculate fps
        fps = fps_calc.update()

        cv2.putText(img_matches, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image with matching scores
        cv2.imshow('Matched Features with Scores', img_matches)
        
        # Keyboard controls to toggle feature detection and matching methods
        key = cv2.waitKey(1) & 0xFF
        # Break the loop on 'q' key press
        if key == ord('q'):
            break
        # switch feature detection method
        elif key == ord('s'):
            feature_method = 'SIFT'
            print("Switched to SIFT")
        # switch feature detection method
        elif key == ord('o'):
            feature_method = 'ORB'
            print("Switched to ORB")
        # switch matching method
        elif key == ord('b'):
            matching_method = 'BF'
            print("Switched to Brute-Force Matching")
        # switch matching method
        elif key == ord('f'):
            matching_method = 'FLANN'
            print("Switched to FLANN-Based Matching")

    # Release the cameras and close all OpenCV windows
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

main()