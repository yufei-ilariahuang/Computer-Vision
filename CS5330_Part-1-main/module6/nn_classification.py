import numpy as np
import pickle
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

def load_cifar10_batch(file):
    """ Load a single batch of CIFAR-10 dataset """
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        X = dict[b'data']
        Y = dict[b'labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y

def load_cifar10(root):
    """ Load all batches of CIFAR-10 dataset """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    X_test, Y_test = load_cifar10_batch(os.path.join(root, 'test_batch'))
    return X_train, Y_train, X_test, Y_test

def euclidean_distance(x1, x2):
    """ Compute the Euclidean distance between two vectors """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def nearest_neighbor_predict(X_train, Y_train, x_test):
    """ Predict the label of a single test example using nearest neighbor """
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    nearest_index = np.argmin(distances)
    return Y_train[nearest_index]

def nearest_neighbor(X_train, Y_train, X_test):
    """ Predict the labels for the test set using nearest neighbor """
    predictions = [nearest_neighbor_predict(X_train, Y_train, x_test) for x_test in X_test]
    return np.array(predictions)

def accuracy(y_true, y_pred):
    """ Calculate the accuracy of the predictions """
    return np.sum(y_true == y_pred) / len(y_true)

# Load CIFAR-10 dataset
cifar10_dir = './cifar-10-batches-py'
X_train, Y_train, X_test, Y_test = load_cifar10(cifar10_dir)

# Flatten the images for simplicity
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Normalize the data
mean = np.mean(X_train_flat, axis=0)
std = np.std(X_train_flat, axis=0)
X_train_flat = (X_train_flat - mean) / std
X_test_flat = (X_test_flat - mean) / std

# Take a subset of the dataset for faster computation (optional)
num_train_samples = 10000
num_test_samples = 2000
indices = random.sample(range(X_train_flat.shape[0]), num_train_samples)
X_train_sub = X_train_flat[indices]
Y_train_sub = Y_train[indices]

indices = random.sample(range(X_test_flat.shape[0]), num_test_samples)
X_test_sub = X_test_flat[indices]
Y_test_sub = Y_test[indices]

# Perform nearest neighbor classification
predictions = nearest_neighbor(X_train_sub, Y_train_sub, X_test_sub)

# Calculate accuracy
acc = accuracy(Y_test_sub, predictions)
print(f'Accuracy: {acc * 100:.2f}%')

# Calculate confusion matrix
cm = confusion_matrix(Y_test_sub, predictions)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.show()
