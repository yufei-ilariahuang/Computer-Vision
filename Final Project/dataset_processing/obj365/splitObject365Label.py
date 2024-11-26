"""
Split the large Object365 labels into separate files.
"""

import gc
import json
import os
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


def process_chunk(chunk_data):
    output_dir, chunk = chunk_data
    for image_data, annotations in chunk:
        # organize the image and annotation data
        image_output = {"image": image_data, "annotations": annotations}

        # generate the output file name
        output_file = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(image_data['file_name']))[0]}.json",
        )

        # save as a separate file
        with open(output_file, "w") as f:
            json.dump(image_output, f)

    # explicitly collect garbage
    gc.collect()
    return len(chunk)


if __name__ == "__main__":
    input_file = "/Users/tongleyao/Downloads/object365/val/sample_2020.json"
    output_dir = "/Users/tongleyao/Downloads/object365/val/origin_labels"
    os.makedirs(output_dir, exist_ok=True)

    # read data in chunks
    chunk_size = 1000  # process 1000 images per process

    print("正在加载数据...")
    with open(input_file, "r") as f:
        data = json.load(f)

    # organize the image information by image id
    images_dict = {image["id"]: image for image in data["images"]}

    # group the annotations by image id
    annotations_by_image = {}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # prepare data chunks
    all_items = [
        (images_dict[image_id], annotations_by_image.get(image_id, []))
        for image_id in images_dict.keys()
    ]
    chunks = [
        all_items[i : i + chunk_size] for i in range(0, len(all_items), chunk_size)
    ]

    # release the memory of the original data
    del data, images_dict, annotations_by_image
    gc.collect()

    # prepare the process pool parameters
    chunk_args = [(output_dir, chunk) for chunk in chunks]
    num_processes = min(
        cpu_count(), len(chunks)
    )  # ensure the number of processes does not exceed the number of chunks

    print(f"start processing with {num_processes} processes...")
    with Pool(num_processes) as pool:
        total_processed = 0
        with tqdm(total=len(all_items), desc="Processing progress") as pbar:
            for n_processed in pool.imap_unordered(process_chunk, chunk_args):
                total_processed += n_processed
                pbar.update(n_processed)
