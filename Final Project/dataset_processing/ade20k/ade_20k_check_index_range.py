label_file_dir = "/Users/tongleyao/Downloads/ADEChallengeData2016/annotations"

import os
from collections import Counter

import cv2
import numpy as np

# get all png files in the directory
png_files = [f for f in os.listdir(label_file_dir) if f.endswith(".png")]
target_pixel = 140

# initialize pixel value counter
pixel_counts = Counter()

# iterate over all PNG files
for png_file in png_files:
    # read image
    img_path = os.path.join(label_file_dir, png_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # check if the image contains the target pixel
    if target_pixel in np.unique(img):
        print(png_file)
        img[img == target_pixel] = 255
        cv2.imshow(png_file, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    # count the number of each pixel value
    counts = Counter(img.ravel())
    pixel_counts.update(counts)

# print the result
print("pixel value count:")
for pixel_value in range(256):
    count = pixel_counts[pixel_value]
    if count > 0:
        print(f"pixel value {pixel_value}: {count} pixels")
