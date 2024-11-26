"""
Create a subset of the Object365 dataset that balances the number of bounding boxes for each class.
"""

import glob
import os
import shutil
from collections import defaultdict

from tqdm import tqdm


def get_annotation_file_paths(annotations_dir):
    annotation_file_paths = glob.glob(os.path.join(annotations_dir, "*.txt"))
    return annotation_file_paths


def count_bboxes(annotation_file_paths):
    label_count = defaultdict(int)
    for annotation_file_path in tqdm(
        annotation_file_paths, desc="counting bounding boxes"
    ):
        with open(annotation_file_path, "r") as file:
            data = file.readlines()
            for line in data:
                label_id = int(line.split()[0])
                label_count[label_id] += 1
    return label_count


def get_label_file_mapping(annotation_file_paths):
    # create a dictionary, key is label_id, value is a dictionary, key is file path, value is the count of bounding boxes for the label
    label_to_files = defaultdict(lambda: defaultdict(int))
    output_dict = {}

    # iterate over all annotation files
    for annotation_file_path in tqdm(
        annotation_file_paths, desc="building label to file mapping"
    ):
        with open(annotation_file_path, "r") as file:
            data = file.readlines()
            # count the number of each label_id in the file
            label_counts = defaultdict(int)
            for line in data:
                label_id = int(line.split()[0])
                label_counts[label_id] += 1
            # add the file path and the count of bounding boxes to the dictionary
            for label_id, count in label_counts.items():
                label_to_files[label_id][annotation_file_path] = count

    # sort the files for each label by the count of bounding boxes
    for label_id, files in label_to_files.items():
        # convert the dictionary to a list of (file_path, count) and sort by count in descending order
        sorted_files = sorted(files.items(), key=lambda x: x[1], reverse=True)
        # only keep the file paths
        output_dict[label_id] = [file_path for file_path, _ in sorted_files]

    return output_dict


# create a balanced dataset
def create_balanced_dataset(label_to_files, label_count, target_bbox_count):
    selected_files = set()
    class_bboxes = defaultdict(int)

    # initialize class_bboxes
    for label_id in label_count.keys():
        class_bboxes[label_id] = 0

    # sort the labels by the count of bounding boxes in ascending order
    sorted_labels = sorted(label_count.items(), key=lambda x: x[1])

    for label_id, _ in tqdm(sorted_labels, desc="processing labels"):
        if class_bboxes[label_id] >= target_bbox_count:
            continue

        for annotation_file_path in tqdm(
            label_to_files[label_id], desc=f"processing label {label_id}"
        ):
            filename = os.path.basename(annotation_file_path)

            # check if the image has been added
            if filename in selected_files:
                continue

            with open(annotation_file_path, "r") as file:
                data = file.readlines()

            # check if the image contains the current label
            has_target_label = any(int(line.split()[0]) == label_id for line in data)
            if not has_target_label:
                continue

            # add the image and update the count of bounding boxes for the label
            selected_files.add(annotation_file_path)
            for line in data:
                current_label = int(line.split()[0])
                class_bboxes[current_label] += 1

            # check if the target count is reached
            if class_bboxes[label_id] >= target_bbox_count:
                break

    return class_bboxes, selected_files


def export_subset(
    selected_annotation_files, images_dir, annotation_output_dir, image_output_dir
):
    image_not_found = []
    # export the subset
    os.makedirs(annotation_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    for annotation_file_path in tqdm(
        selected_annotation_files, desc="exporting subset"
    ):
        shutil.copy(
            annotation_file_path,
            os.path.join(annotation_output_dir, os.path.basename(annotation_file_path)),
        )
        image_file_path = os.path.join(
            images_dir, os.path.basename(annotation_file_path).replace(".txt", ".jpg")
        )
        if os.path.exists(image_file_path):
            shutil.copy(
                image_file_path,
                os.path.join(image_output_dir, os.path.basename(image_file_path)),
            )
        else:
            image_not_found.append(image_file_path)

    print(f"image not found: {image_not_found}")
