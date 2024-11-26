"""
This script is used to split the database into train, val, and test sets.
expect the file structure is as follows:
/dataset_root
    /images
        image files has the same name as the label file
    /labels
        yolo_format txt label files
"""

import argparse
import os

from dataset_processing.split_dataset import split_dataset
from dataset_processing.subset_balance_label_bbox import (
    count_bboxes,
    create_balanced_dataset,
    export_subset,
    get_annotation_file_paths,
    get_label_file_mapping,
)


def generate_balanced_dataset(dataset_root, target_bbox_count):
    """
    Generate a balanced dataset by selecting numbers of bounding boxes for each class from the full dataset images
    """
    annotations_dir = os.path.join(dataset_root, "labels")
    os.makedirs(annotations_dir, exist_ok=True)
    images_dir = os.path.join(dataset_root, "images")
    os.makedirs(images_dir, exist_ok=True)
    balanced_dataset_root = os.path.join(dataset_root, f"balanced_{target_bbox_count}")
    os.makedirs(balanced_dataset_root, exist_ok=True)
    balanced_annot_output_dir = os.path.join(balanced_dataset_root, "labels")
    os.makedirs(balanced_annot_output_dir, exist_ok=True)
    balanced_images_output_dir = os.path.join(balanced_dataset_root, "images")
    os.makedirs(balanced_images_output_dir, exist_ok=True)

    print("processing dataset...")
    annotation_file_paths = get_annotation_file_paths(annotations_dir)
    print("counting bounding boxes...")
    label_count = count_bboxes(annotation_file_paths)
    print("building label to file mapping...")
    label_to_files = get_label_file_mapping(annotation_file_paths)
    print("creating balanced dataset...")
    balanced_counts, selected_annotation_files = create_balanced_dataset(
        label_to_files, label_count, target_bbox_count
    )
    print(
        f"balanced counts: {balanced_counts}\nselected {len(selected_annotation_files)} annotation files\nexporting subset..."
    )
    export_subset(
        selected_annotation_files,
        images_dir,
        balanced_annot_output_dir,
        balanced_images_output_dir,
    )

    return balanced_annot_output_dir, balanced_images_output_dir


def split_database(
    dataset_img_dir, dataset_annot_dir, output_root, val_ratio, test_ratio
):
    output_dir = os.path.join(output_root, f"balanced_{val_ratio}_{test_ratio}")
    annot_output_root = os.path.join(output_dir, "labels")
    os.makedirs(annot_output_root, exist_ok=True)
    image_output_root = os.path.join(output_dir, "images")
    os.makedirs(image_output_root, exist_ok=True)

    print("splitting dataset...")
    split_dataset(
        dataset_img_dir,
        dataset_annot_dir,
        annot_output_root,
        image_output_root,
        val_ratio,
        test_ratio,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the database into train, val, and test sets."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="The root directory of the full dataset.",
    )
    parser.add_argument(
        "--target_bbox",
        type=int,
        required=True,
        help="The target bounding box count for each class.",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        required=True,
        help="The ratio of the validation set.",
    )
    parser.add_argument(
        "--test_ratio", type=float, required=True, help="The ratio of the test set."
    )

    args = parser.parse_args()
    dataset_root = args.dataset_root
    target_bbox_count = args.target_bbox
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio

    balanced_annot_output_dir, balanced_images_output_dir = generate_balanced_dataset(
        dataset_root, target_bbox_count
    )
    split_database(
        balanced_images_output_dir,
        balanced_annot_output_dir,
        dataset_root,
        val_ratio,
        test_ratio,
    )
