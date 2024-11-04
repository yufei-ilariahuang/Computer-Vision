# Image Stitcher

## Overview

This project provides a robust command-line tool for stitching multiple images together, specifically designed to work with sets of 4 or 9 images. It's ideal for creating panoramas or combining fragmented images into a cohesive whole.

## Features

- Stitches 2x2 or 3x3 grids of images seamlessly
- Utilizes OpenCV for advanced image processing and feature matching
- Implements SIFT (Scale-Invariant Feature Transform) for keypoint detection
- Uses FLANN (Fast Library for Approximate Nearest Neighbors) for efficient feature matching
- Automatically determines optimal image positions in the grid
- Handles both vertical and horizontal stitching with fallback options
- Post-processes results to remove black borders and enhance contrast

## Requirements

- Python 3.6+
- OpenCV 4.5+
- NumPy 1.19+

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/image-stitcher.git
   cd image-stitcher
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate image_stitcher_env
   ```

## Usage

Run the script from the command line, providing the path to the folder containing your images:

```bash
python tile-quiz.py path/to/your/image/folder
```

### Example

```bash
python tile-quiz.py ./data/set1
```

This will process all images in the `./data/set1` folder and output a `stitched_result.jpg` in the same directory.

## Input Requirements

- The input folder should contain either 4 or 9 images
- Images should be in .jpg, .jpeg, or .png format
- Images should have sufficient overlap for feature matching
- Filenames should be in alphabetical or numerical order corresponding to their position in the grid (e.g., top-left, top-middle, top-right, etc.)

## Output

- The stitched image will be saved as `stitched_result.jpg` in the input folder
- A preview of the result will be displayed on screen

## Algorithm Overview

1. **Feature Detection**: Uses SIFT to identify key features in each image
2. **Feature Matching**: Employs FLANN to match features between image pairs
3. **Homography Estimation**: Calculates transformation matrices using RANSAC
4. **Image Warping**: Applies perspective transformations to align images
5. **Blending**: Implements multi-band blending for seamless transitions
6. **Post-processing**: Crops black borders and enhances contrast of the final image

## Troubleshooting

If you encounter issues with stitching:
- Ensure your images have at least 30% overlap
- Try adjusting the `match_ratio` parameter in `ImageStitcher` class (default is 0.7)
- Check console output for detailed logging of the stitching process
- For low-contrast images, consider pre-processing to enhance features

## Examples

### Input Images
![Input Images](path/to/input_images_example.jpg)

### Stitched Result
![Stitched Result](path/to/stitched_result_example.jpg)

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/image-stitcher/issues).

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- OpenCV community for their comprehensive computer vision library
- David Lowe for the SIFT algorithm
- [Any other acknowledgments or references]

## Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/yourusername/image-stitcher](https://github.com/yourusername/image-stitcher)