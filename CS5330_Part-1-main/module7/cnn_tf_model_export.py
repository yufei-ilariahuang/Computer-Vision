import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam
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

# Define the CNN architecture
model = Sequential()
model.add(Conv2D(8, (3, 3), padding="same", input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(len(lb.classes_)))
model.add(Activation("softmax"))

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

# Export the model
print("[INFO] saving model...")
model.save('scene_classifier_model.h5')  # Save model in HDF5 format
