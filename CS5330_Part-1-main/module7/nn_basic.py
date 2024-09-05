import numpy as np
import cv2
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

# Flatten the images for the neural network input
trainX = trainX.reshape(trainX.shape[0], -1)
testX = testX.reshape(testX.shape[0], -1)

# Neural network architecture
input_size = 32 * 32 * 3
hidden_size = 64
output_size = len(lb.classes_)
learning_rate = 0.01
epochs = 50
batch_size = 32

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Training the neural network
for epoch in range(epochs):
    # Shuffle the training data
    indices = np.arange(trainX.shape[0])
    np.random.shuffle(indices)
    trainX = trainX[indices]
    trainY = trainY[indices]

    # Mini-batch gradient descent
    for i in range(0, trainX.shape[0], batch_size):
        end = i + batch_size
        batchX = trainX[i:end]
        batchY = trainY[i:end]

        # Forward pass
        z1 = np.dot(batchX, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)

        # Compute loss (cross-entropy)
        loss = -np.mean(np.sum(batchY * np.log(a2 + 1e-8), axis=1))

        # Backward pass
        dz2 = a2 - batchY
        dW2 = np.dot(a1.T, dz2) / batchX.shape[0]
        db2 = np.sum(dz2, axis=0, keepdims=True) / batchX.shape[0]
        
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(z1)
        dW1 = np.dot(batchX.T, dz1) / batchX.shape[0]
        db1 = np.sum(dz1, axis=0, keepdims=True) / batchX.shape[0]

        # Update weights and biases
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

# Testing the neural network
z1 = np.dot(testX, W1) + b1
a1 = relu(z1)
z2 = np.dot(a1, W2) + b2
a2 = softmax(z2)

# Predictions
predictions = np.argmax(a2, axis=1)
labels = np.argmax(testY, axis=1)

# Accuracy and classification report
accuracy = np.mean(predictions == labels)
print(f'Test accuracy: {accuracy * 100:.2f}%')
print(classification_report(labels, predictions, target_names=lb.classes_))

# Display 3 sample test images with predictions
sample_indices = np.random.choice(len(testX), 3, replace=False)
sample_images = testX[sample_indices].reshape(-1, 32, 32, 3)
sample_predictions = predictions[sample_indices]
sample_labels = labels[sample_indices]

for i, index in enumerate(sample_indices):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(sample_images[i], cv2.COLOR_BGR2RGB))
    plt.title(f"Pred: {lb.classes_[sample_predictions[i]]}\nTrue: {lb.classes_[sample_labels[i]]}")
    plt.axis('off')

plt.show()
