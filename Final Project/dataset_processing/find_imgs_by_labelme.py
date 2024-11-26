"""
Find the images based on the labelme json files and move them to the target directory.
"""

import glob
import json
import os
import shutil

from tqdm import tqdm


def extract_img_path_from_labelme_json(labelme_json_path):
    with open(labelme_json_path, "r") as f:
        data = json.load(f)
    relative_img_path = data["imagePath"]
    json_dir = os.path.dirname(labelme_json_path)
    return os.path.join(json_dir, relative_img_path)


def move_img_file_to_dir(img_path, target_dir):
    new_img_path = os.path.join(target_dir, os.path.basename(img_path))
    shutil.move(img_path, new_img_path)
    return new_img_path


def update_labelme_json(labelme_json_path, data, updated_img_path):
    # generate relative path
    json_dir = os.path.dirname(labelme_json_path)
    abs_img_path = os.path.abspath(updated_img_path)
    abs_json_dir = os.path.abspath(json_dir)
    rel_path = os.path.relpath(abs_img_path, abs_json_dir)
    updated_img_path = rel_path.replace("\\", "/")

    data["imagePath"] = updated_img_path
    with open(labelme_json_path, "w") as f:
        json.dump(data, f)


def main():
    labelme_json_dir = "/Users/tongleyao/Downloads/object365/train/labelme"
    filtered_img_dir = "/Users/tongleyao/Downloads/object365/train/images"
    os.makedirs(filtered_img_dir, exist_ok=True)

    labelme_json_files = glob.glob(os.path.join(labelme_json_dir, "*.json"))

    labelme_json_file = labelme_json_files[0]
    failed_labelme_json_files = []

    for labelme_json_file in tqdm(labelme_json_files):
        img_path = extract_img_path_from_labelme_json(labelme_json_file)

        with open(labelme_json_file, "r") as f:
            data = json.load(f)
        if os.path.exists(img_path):
            new_img_path = move_img_file_to_dir(img_path, filtered_img_dir)
            update_labelme_json(labelme_json_file, data, new_img_path)
        else:
            failed_labelme_json_files.append(labelme_json_file)

    print(f"Failed labelme jsons: {failed_labelme_json_files}")


if __name__ == "__main__":
    main()
