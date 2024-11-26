import json
import os
from collections import defaultdict

import cv2
import numpy as np


def save_labelme_file(labelme_format, file_path):
    with open(file_path, "w") as f:
        json.dump(labelme_format, f, indent=4)


def get_shapes(labelme_format, bounding_boxes):
    shapes = []
    for cls_id, boxes in bounding_boxes.items():
        for box in boxes:
            shape = {
                "label": (
                    idx_to_class_name[str(cls_id)]
                    if isinstance(idx_to_class_name[str(cls_id)], str)
                    else idx_to_class_name[str(cls_id)]["name"]
                ),
                "points": [],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
            shape["points"].append([int(box[0][0]), int(box[0][1])])
            shape["points"].append([int(box[1][0]), int(box[1][1])])
            shapes.append(shape)
    labelme_format["shapes"] = shapes


def get_bounding_box(label_image):
    bounding_boxes = defaultdict(list)

    for cls_id in idx_to_class_name.keys():
        cls_id = int(cls_id)
        value = idx_to_class_name[str(cls_id)]
        mapping_id = []
        if isinstance(value, dict):
            mapping_id = value["mapping_id"]

        # create a binary mask image, for ids in mapping_id, consider them as the same class as cls_id
        binary_mask = (label_image == cls_id).astype(np.uint8)
        for id in mapping_id:
            binary_mask[label_image == id] = 1

        # find all connected components (objects)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        # extract bounding boxes for each connected component
        for i in range(1, num_labels):  # start from 1, skip background 0
            x, y, w, h, area = stats[i]

            # calculate the top-left and bottom-right coordinates
            x1, y1 = x, y
            x2, y2 = x + w - 1, y + h - 1

            # save the bounding box, format as [x1, y1, x2, y2]
            bounding_boxes[cls_id].append(((x1, y1), (x2, y2), area))

        for cls_id, boxes in bounding_boxes.items():
            if len(boxes) < 2:
                continue

            max_area = max([box[2] for box in boxes])

            # collect bounding boxes that are less than 30% of the maximum area
            to_remove = [box for box in boxes if box[2] < 0.3 * max_area]

            # remove the bounding boxes
            for box in to_remove:
                bounding_boxes[cls_id].remove(box)

    return bounding_boxes


label_file_dir = "/Users/tongleyao/Downloads/ADEChallengeData2016/annotations"
label_file_names = [f for f in os.listdir(label_file_dir) if f.endswith(".png")]

for label_file_name in label_file_names:
    output_file_dir = os.path.join(os.path.dirname(label_file_dir), "labelme")

    label_image = cv2.imread(
        os.path.join(label_file_dir, label_file_name), cv2.IMREAD_GRAYSCALE
    )
    labelme_format = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.join(
            "..", "images", os.path.basename(label_file_name).replace(".png", ".jpg")
        ),
        "imageData": None,
        "imageHeight": label_image.shape[0],
        "imageWidth": label_image.shape[1],
    }
    idx_to_class_name = json.load(
        open("/Users/tongleyao/Downloads/ADEChallengeData2016/objectInfo150.json", "r")
    )

    bounding_boxes = get_bounding_box(label_image)
    if len(bounding_boxes) > 0:
        get_shapes(labelme_format, bounding_boxes)
        save_labelme_file(
            labelme_format,
            os.path.join(output_file_dir, label_file_name.replace(".png", ".json")),
        )
