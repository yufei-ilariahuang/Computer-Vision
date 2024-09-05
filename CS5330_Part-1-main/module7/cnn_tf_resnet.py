import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import argparse
import glob

# Define the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
                help="path to directory containing the '3scenes' dataset")
args = vars(ap.parse_args())

# Grab all image paths in the input dataset directory
print("[INFO] loading images...")
imagePaths = glob.glob(os.path.join(args["dataset"], "**", "*.jpg"), recursive=True)
data = []
labels = []

# Load and preprocess the images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    image = image.astype("float32") / 255.0
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# Encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Perform a training and testing split
(trainX, testX, trainY, testY) = train_test_split(np.array(data), np.array(labels), test_size=0.25)

# Define the ResNet architecture
baseModel = ResNet50(weights="imagenet", include_top=False, input_shape=(32, 32, 3))

# Build the model
model = Sequential()
model.add(baseModel)
model.add(GlobalAveragePooling2D())
model.add(Dense(len(lb.classes_), activation="softmax"))

# Freeze the base model layers to prevent them from being updated during training
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] compiling model...")
opt = Adam(lr=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=32)

# Evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# Display 3 sample test images with predictions
sample_indices = np.random.choice(len(testX), 3, replace=False)
sample_images = testX[sample_indices]
sample_predictions = predictions[sample_indices]
sample_labels = testY[sample_indices]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(sample_images[i], cv2.COLOR_BGR2RGB))
    plt.title(f"Pred: {lb.classes_[sample_predictions[i].argmax()]}\nTrue: {lb.classes_[sample_labels[i].argmax()]}")
    plt.axis('off')

plt.show()
