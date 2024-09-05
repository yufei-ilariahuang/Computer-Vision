import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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

# Define a custom dataset class for PyTorch
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].transpose(2, 0, 1)  # Convert to CHW format
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Create DataLoader objects for training and testing
train_dataset = CustomDataset(trainX, trainY)
test_dataset = CustomDataset(testX, testY)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = torch.softmax(self.fc1(x), dim=1)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleCNN(num_classes=len(lb.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
print("[INFO] training network...")
for epoch in range(50):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/50, Loss: {total_loss/len(train_loader)}')

# Evaluating the model
print("[INFO] evaluating network...")
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.argmax(dim=1).cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=lb.classes_))

# Display 3 sample test images with predictions
sample_indices = np.random.choice(len(testX), 3, replace=False)
sample_images = testX[sample_indices]
sample_labels = testY[sample_indices]
sample_preds = []

model.eval()
with torch.no_grad():
    for image in sample_images:
        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        output = model(image_tensor)
        pred = output.argmax(dim=1).item()
        sample_preds.append(pred)

for i, index in enumerate(sample_indices):
    plt.subplot(1, 3, i + 1)
    plt.imshow(cv2.cvtColor(sample_images[i], cv2.COLOR_BGR2RGB))
    plt.title(f"Pred: {lb.classes_[sample_preds[i]]}\nTrue: {lb.classes_[sample_labels[i].argmax()]}")
    plt.axis('off')

plt.show()
