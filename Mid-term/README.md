# CS5330-Mid-Term-Fall2024 - Group 5

## Group Member
- Siyi Gao
- Yufei Huang
- Xuedong Pan
- Tongle Yao

## Project Description
This project performs image stitching using feature matching (SIFT and FLANN matcher) and homography estimation. The program reads four images from a directory, determines their relative positions, and stitches them into a final, combined image. The images are assumed to be parts of a larger whole, split into a 2x2 grid. After detecting features and keypoints in the images, the program stitches them row by row and then stitches the rows vertically to form the final image. SIFT is used to detect and compute keypoints and descriptors in the images, and the FLANN-based matcher is used to find corresponding points between images, which are then used to compute a homography matrix for image alignment. The final stitched image is post-processed to remove any black borders resulting from the transformations.

## Requirements
- Python
- OpenCV
- NumPy

## Code Explanation

### __init__:
   - initialize ImageStitcher class, set the response threshold and match ratio.
   - create SIFT (Scale-Invariant Feature Transform) feature detector.
   - configure FLANN (Fast Library for Approximate Nearest Neighbors) matcher, used for fast feature matching.

### determine_image_positions:
   - automatically determine the position of each image in a 2x2 grid. This is achieved through the following steps:
     1. feature extraction on all image pairs, using SIFT algorithm to detect keypoints and compute descriptors.
     2. use FLANN matcher to match features between image pairs.
     3. apply ratio test to filter out high-quality matches.
     4. calculate the average relative displacement between matches, including horizontal and vertical directions.
     5. infer the relative position relationship of images in the 2x2 grid based on the size and direction of displacement.
     6. finally, determine the position of each image should be in the top-left, top-right, bottom-left, or bottom-right position.

### stitch_pair:
   - stitch two images. This process includes the following steps:
     1. convert two input images to grayscale.
     2. use SIFT algorithm to detect keypoints and compute descriptors on two images.
     3. use FLANN matcher to match features between two images.
     4. apply ratio test to filter out high-quality matches.
     5. if the number of matches is sufficient, use RANSAC algorithm to compute homography matrix.
     6. according to homography matrix, perform perspective transformation on the second image.
     7. create a large canvas that can contain two images.
     8. place the first image on the canvas.
     9. transform the second image and merge it with the first image using alpha blending or other fusion techniques.
     10. return the stitched result image.
   
### post_process 方法:
   - post-process the stitched image. The specific steps are as follows:
     1. convert the stitched color image to grayscale.
     2. perform thresholding on the grayscale image to binarize the non-black regions.
     3. find contours in the binarized image.
     4. select the contour with the largest area, which is considered the main content area.
     5. calculate the bounding rectangle of this contour.
     6. use the coordinates of the bounding rectangle to crop the original color image, removing the black borders.
     7. optionally perform contrast enhancement on the cropped image to improve visual quality.

### stitch_images:
   - stitch images according to the number of images.
   - if the number of images is 4, call stitch_four_images method.
   - if the number of images is 9, call stitch_nine_images method.  


## Usage

```bash
python tile-quiz.py -d <data-folder>
```

## Example

## data-1 4 images 
<img src="data-1_4.jpg" alt="result" width="500">

## data-2 4 images
<img src="data-2_4.jpg" alt="result" width="500">

## data-3 4 images
<img src="data-3_4.jpg" alt="result" width="500">

## data-1 9 images
<img src="data-1_9.jpg" alt="result" width="500">

## data-2 9 images
<img src="data-2_9.jpg" alt="result" width="500">
