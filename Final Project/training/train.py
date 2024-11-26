import argparse

import numpy as np
import torch.backends.cudnn as cudnn
import ultralytics.data.build as build
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from utils import get_device


class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
                weights.append(1)
                continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))


def train(config_path, imgsz=640):
    build.YOLODataset = YOLOWeightedDataset

    # load the model
    model = YOLO("yolo11s.pt")
    cudnn.benchmark = True
    # train the model
    model.train(
        data=config_path,
        epochs=200,
        imgsz=imgsz,
        device=get_device(),
        augment=True,
        amp=True,
        patience=20,
        cache=True,
        workers=8,
        batch=0.9,
        dropout=0.2,
        plots=True,
        degrees=90.0,
        mosaic=0.7,
        # lrf=0.001
    )
    return model


def export(model):
    # export the model to ONNX format
    path = model.export(format="onnx")  # return the path of the exported model


def main(config_path):
    model = train(config_path)
    export(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="dataset config path")
    parser.add_argument("--imgsz", type=int, required=False, default=640, help="image size")
    args = parser.parse_args()
    config_path = args.config
    imgsz = args.imgsz
    main(config_path, imgsz)
