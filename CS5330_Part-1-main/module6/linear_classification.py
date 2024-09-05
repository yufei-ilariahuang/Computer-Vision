import numpy as np
import pickle
import os
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

def softmax(z):
    """ Compute the softmax of each row of the input array. """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """ Compute the cross-entropy loss """
    n = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true])
    loss = np.sum(log_likelihood) / n
    return loss

def compute_accuracy(y_true, y_pred):
    """ Compute the accuracy of the predictions """
    return np.mean(y_true == y_pred)

def linear_classifier(X_train, y_train, X_test, y_test, learning_rate=0.1, epochs=100, batch_size=100):
    """ Train a linear classifier using gradient descent """
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros((1, n_classes))

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            scores = np.dot(X_batch, W) + b
            probs = softmax(scores)
            loss = cross_entropy_loss(y_batch, probs)

            # Compute the gradients
            dscores = probs
            dscores[range(batch_size), y_batch] -= 1
            dscores /= batch_size

            dW = np.dot(X_batch.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)

            # Update the weights and biases
            W -= learning_rate * dW
            b -= learning_rate * db

        if epoch % 10 == 0:
            train_scores = np.dot(X_train, W) + b
            train_preds = np.argmax(train_scores, axis=1)
            train_acc = compute_accuracy(y_train, train_preds)
            print(f'Epoch {epoch}, Loss: {loss:.4f}, Training Accuracy: {train_acc:.4f}')

    # Evaluate on test data
    test_scores = np.dot(X_test, W) + b
    test_preds = np.argmax(test_scores, axis=1)
    test_acc = compute_accuracy(y_test, test_preds)
    return W, b, test_preds, test_acc

# Load CIFAR-10 dataset
cifar10_dir = 'cifar-10-batches-py'
X_train, Y_train, X_test, Y_test = load_cifar10(cifar10_dir)

# Flatten the images for simplicity
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Normalize the data
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# Take a subset of the dataset for faster computation (optional)
num_train_samples = 10000
num_test_samples = 200
indices = random.sample(range(X_train_flat.shape[0]), num_train_samples)
X_train_sub = X_train_flat[indices]
Y_train_sub = Y_train[indices]

indices = random.sample(range(X_test_flat.shape[0]), num_test_samples)
X_test_sub = X_test_flat[indices]
Y_test_sub = Y_test[indices]

# Train Linear classifier
learning_rate = 0.1
epochs = 100
batch_size = 100
W, b, test_preds, test_acc = linear_classifier(X_train_sub, Y_train_sub, X_test_sub, Y_test_sub, learning_rate, epochs, batch_size)

# Calculate accuracy
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Calculate precision and recall
precision = precision_score(Y_test_sub, test_preds, average='macro')
recall = recall_score(Y_test_sub, test_preds, average='macro')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')

# Calculate confusion matrix
cm = confusion_matrix(Y_test_sub, test_preds)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.show()

