import os
import shutil

from tqdm import tqdm


def read_labels(annot_dir):
    # store the file name and bbox count for each class
    class_stats = {}  # {class_id: [(file_name, bbox_count), ...]}

    # iterate through all label files
    for label_file in tqdm(os.listdir(annot_dir), desc="Reading labels"):
        if not label_file.endswith(".txt"):
            continue

        file_path = os.path.join(annot_dir, label_file)
        with open(file_path, "r") as f:
            lines = f.readlines()

        # count the bbox number for each class in the file
        file_class_count = {}  # {class_id: count}
        for line in lines:
            class_id = int(line.split()[0])
            if class_id not in file_class_count:
                file_class_count[class_id] = 0
            file_class_count[class_id] += 1

        # update the total statistics
        for class_id, count in file_class_count.items():
            if class_id not in class_stats:
                class_stats[class_id] = []
            class_stats[class_id].append((label_file, count))

    return class_stats


def check_output_dir(
    train_annot_output_dir,
    train_img_output_dir,
    val_annot_output_dir,
    val_img_output_dir,
    test_annot_output_dir,
    test_img_output_dir,
):
    os.makedirs(train_annot_output_dir, exist_ok=True)
    os.makedirs(train_img_output_dir, exist_ok=True)
    os.makedirs(val_annot_output_dir, exist_ok=True)
    os.makedirs(val_img_output_dir, exist_ok=True)
    os.makedirs(test_annot_output_dir, exist_ok=True)
    os.makedirs(test_img_output_dir, exist_ok=True)


def split_dataset(
    img_dir, annot_dir, annot_output_root, image_output_root, val_ratio, test_ratio
):
    train_annot_output_dir = os.path.join(annot_output_root, "train")
    train_img_output_dir = os.path.join(image_output_root, "train")
    val_annot_output_dir = os.path.join(annot_output_root, "val")
    val_img_output_dir = os.path.join(image_output_root, "val")
    test_annot_output_dir = os.path.join(annot_output_root, "test")
    test_img_output_dir = os.path.join(image_output_root, "test")

    check_output_dir(
        train_annot_output_dir,
        train_img_output_dir,
        val_annot_output_dir,
        val_img_output_dir,
        test_annot_output_dir,
        test_img_output_dir,
    )
    class_stats = read_labels(annot_dir)

    # calculate the total bbox number for each class and sort them
    class_total_boxes = {
        class_id: sum(count for _, count in files)
        for class_id, files in class_stats.items()
    }
    sorted_classes = sorted(class_total_boxes.items(), key=lambda x: x[1])

    # track the files that have been assigned
    assigned_files = set()
    val_files = set()
    test_files = set()

    # process each class
    for class_id, total_boxes in sorted_classes:
        class_files = class_stats[class_id]

        # calculate the bbox number needed for val and test
        val_target = int(total_boxes * val_ratio)
        test_target = int(total_boxes * test_ratio)

        # count the bbox number in the files that have been assigned for the current class
        current_val_boxes = sum(
            count for filename, count in class_files if filename in val_files
        )
        current_test_boxes = sum(
            count for filename, count in class_files if filename in test_files
        )

        # add more val files
        for filename, count in class_files:
            if current_val_boxes >= val_target:
                break
            if filename not in assigned_files:
                val_files.add(filename)
                assigned_files.add(filename)
                current_val_boxes += count

        # add more test files
        for filename, count in class_files:
            if current_test_boxes >= test_target:
                break
            if filename not in assigned_files:
                test_files.add(filename)
                assigned_files.add(filename)
                current_test_boxes += count

    # get all unique file names
    all_files = set()
    for files in class_stats.values():
        all_files.update(f for f, _ in files)

    # the remaining files are assigned to the training set
    train_files = all_files - assigned_files
    print(f"train files: {len(train_files)}")
    print(f"val files: {len(val_files)}")
    print(f"test files: {len(test_files)}")
    image_not_found = []

    # copy files to the corresponding directories
    for filename in tqdm(train_files, desc="Copying train files"):
        base_name = os.path.splitext(filename)[0]
        image_file_path = os.path.join(img_dir, f"{base_name}.jpg")
        copy_image_file_path = os.path.join(train_img_output_dir, f"{base_name}.jpg")

        shutil.copy(
            os.path.join(annot_dir, filename),
            os.path.join(train_annot_output_dir, filename),
        )
        if os.path.exists(image_file_path):
            shutil.copy(image_file_path, copy_image_file_path)
        else:
            image_not_found.append(image_file_path)

    for filename in tqdm(val_files, desc="Copying validation files"):
        base_name = os.path.splitext(filename)[0]
        image_file_path = os.path.join(img_dir, f"{base_name}.jpg")
        copy_image_file_path = os.path.join(val_img_output_dir, f"{base_name}.jpg")

        shutil.copy(
            os.path.join(annot_dir, filename),
            os.path.join(val_annot_output_dir, filename),
        )

        if os.path.exists(image_file_path):
            shutil.copy(image_file_path, copy_image_file_path)
        else:
            image_not_found.append(image_file_path)

    for filename in tqdm(test_files, desc="Copying test files"):
        base_name = os.path.splitext(filename)[0]
        image_file_path = os.path.join(img_dir, f"{base_name}.jpg")
        copy_image_file_path = os.path.join(test_img_output_dir, f"{base_name}.jpg")

        shutil.copy(
            os.path.join(annot_dir, filename),
            os.path.join(test_annot_output_dir, filename),
        )

        if os.path.exists(image_file_path):
            shutil.copy(image_file_path, copy_image_file_path)
        else:
            image_not_found.append(image_file_path)

    print(f"image not found: {image_not_found}")
