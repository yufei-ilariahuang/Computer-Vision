"""
Filter the labels of the Object365 dataset by custom definition. 
Remove the labels that are not in the custom definition.
Convert the filtered labels file to LabelMe format.
"""

import gc
import json
import os
import re
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def struct_label_map(label_map):
    new_label_map = {}

    for k, v in label_map.items():
        if isinstance(v, str):
            new_label_map[int(k)] = v
        elif isinstance(v, dict):
            name = v["name"]
            new_label_map[int(k)] = name
            for mapping_id in v["mapping_ids"]:
                new_label_map[int(mapping_id)] = name
        else:
            raise ValueError(f"Unknown label type: {type(v)}")

    return new_label_map


def filter_annotations(annotations, label_map):
    valid_annotations = []
    labels = set()
    for annotation in annotations:
        if annotation["category_id"] not in label_map:
            continue
        valid_annotations.append(annotation)
        labels.add(annotation["category_id"])

    if len(labels) == 1:
        label = label_map[labels.pop()]
        if label in [
            "Person",
            "car",
            "Cabinet/shelf",
            "Chair",
            "Street Lights",
            "Desk",
        ]:
            return []
    return valid_annotations


def get_corner_coords(annotation):
    bbox = annotation["bbox"]
    left_top = (bbox[0], bbox[1])
    right_bottom = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    return left_top, right_bottom


def format_labelme_data(height, width, image_file_path, annotations, label_map):
    pattern = "patch\d+\/[a-z\d_]+.jpg"
    image_file_path = re.search(pattern, image_file_path).group(0)

    labelme_format = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.join("..", "images", image_file_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

    for annotation in annotations:
        left_top, right_bottom = get_corner_coords(annotation)
        shape = {
            "label": label_map[annotation["category_id"]],
            "points": [left_top, right_bottom],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
        }
        labelme_format["shapes"].append(shape)

    return labelme_format


def save_labelme_data(labelme_format, output_dir, image_file_path):
    with open(
        os.path.join(
            output_dir, os.path.basename(image_file_path).replace(".jpg", ".json")
        ),
        "w",
    ) as f:
        json.dump(labelme_format, f)


def remove_deleted_files(deleted_file_paths):
    for file_path in tqdm(deleted_file_paths, desc="file deleted"):
        os.remove(file_path)


def process_single_file(args):
    # unpack the arguments
    file, label_dir, output_dir, label_map = args
    try:
        with open(os.path.join(label_dir, file), "r") as f:
            label_data = json.load(f)

        annotations = label_data["annotations"]
        image_data = label_data["image"]

        annotations = filter_annotations(annotations, label_map)
        if len(annotations) == 0:
            return file, True  # mark to delete

        labelme_format = format_labelme_data(
            image_data["height"],
            image_data["width"],
            image_data["file_name"],
            annotations,
            label_map,
        )
        save_labelme_data(labelme_format, output_dir, image_data["file_name"])

        # clean up memory
        del label_data, annotations, labelme_format
        gc.collect()

        return file, False  # mark not to delete
    except Exception as e:
        print(f"error processing {file}: {str(e)}")
        return file, False


def main():
    label_map_file = "/Users/tongleyao/Downloads/object365/object365.json"
    label_dir = "/Users/tongleyao/Downloads/object365/val/labels"
    output_dir = "/Users/tongleyao/Downloads/object365/train/labelme"

    os.makedirs(output_dir, exist_ok=True)

    with open(label_map_file, "r") as f:
        label_map = json.load(f)

    label_map = struct_label_map(label_map)
    files = os.listdir(label_dir)

    # create a process pool
    num_processes = cpu_count()
    pool = Pool(processes=num_processes)

    # prepare task parameters
    tasks = [(file, label_dir, output_dir, label_map) for file in files]

    # use tqdm to show progress and execute multiprocessing
    deleted_file_paths = []
    with tqdm(total=len(files), desc="processing files") as pbar:
        for file, should_delete in pool.imap_unordered(process_single_file, tasks):
            if should_delete:
                deleted_file_paths.append(os.path.join(label_dir, file))
            pbar.update(1)

    pool.close()
    pool.join()

    remove_deleted_files(deleted_file_paths)


if __name__ == "__main__":
    main()
