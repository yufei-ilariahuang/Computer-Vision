"""
Count the number of bounding boxes for each label in the labelme json files.
"""

import json
import os
from functools import partial
from multiprocessing import Manager, Pool


def count_labels(file_list, label_dir, shared_dict):
    local_count = {}

    for file in file_list:
        with open(os.path.join(label_dir, file), "r") as f:
            data = json.load(f)
            for shape in data["shapes"]:
                local_count[shape["label"]] = local_count.get(shape["label"], 0) + 1

    # update the shared dictionary
    for label, count in local_count.items():
        if label in shared_dict:
            shared_dict[label] += count
        else:
            shared_dict[label] = count


def count_file_one_label(file_list, label_dir, shared_dict):
    one_label_count = {}

    for file in file_list:
        with open(os.path.join(label_dir, file), "r") as f:
            data = json.load(f)
            # get all the labels of the shapes
            labels = [shape["label"] for shape in data["shapes"]]
            # if all the labels are the same
            if len(set(labels)) == 1 and len(labels) > 0:
                label = labels[0]
                one_label_count[label] = one_label_count.get(label, 0) + 1

    # update the shared dictionary
    for label, count in one_label_count.items():
        shared_dict[label] = shared_dict.get(label, 0) + count


def count_bounding_box_one_label(file_list, label_dir, shared_dict):
    one_label_count = {}

    for file in file_list:
        with open(os.path.join(label_dir, file), "r") as f:
            data = json.load(f)
            # get all the labels of the shapes
            labels = [shape["label"] for shape in data["shapes"]]
            # if all the labels are the same
            if len(set(labels)) == 1 and len(labels) > 0:
                label = labels[0]
                one_label_count[label] = one_label_count.get(label, 0) + len(
                    data["shapes"]
                )

    # update the shared dictionary
    for label, count in one_label_count.items():
        shared_dict[label] = shared_dict.get(label, 0) + count


def main():
    label_dir = "/Users/tongleyao/Downloads/object365/train/labelme"

    # get all the file list
    all_files = os.listdir(label_dir)

    # create the number of processes based on the CPU cores
    num_processes = os.cpu_count()

    # calculate the number of files each process will handle
    chunk_size = len(all_files) // num_processes

    # split the file list into chunks
    file_chunks = [
        all_files[i : i + chunk_size] for i in range(0, len(all_files), chunk_size)
    ]

    # create a shared dictionary
    manager = Manager()
    shared_dict = manager.dict()

    # create a process pool
    with Pool(num_processes) as pool:
        # use partial to fix the label_dir and shared_dict parameters
        count_func = partial(
            count_bounding_box_one_label, label_dir=label_dir, shared_dict=shared_dict
        )
        # start multiprocessing
        pool.map(count_func, file_chunks)

    # convert to a normal dictionary and print the result
    result = dict(shared_dict)
    print(result)


if __name__ == "__main__":
    main()
