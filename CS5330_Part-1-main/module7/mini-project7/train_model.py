import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import argparse
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# data enhancement and normalization
def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(
            degrees=180,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

# split dataset
def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    return train_set, val_set, test_set

# create data loader
def get_data_loaders(data_dir, batch_size=32):
    train_transform, val_test_transform = get_data_transforms()
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)  # ImageFolder will automatically label the images and apply the image enhancement
    
    # split the original dataset into training, validation and testing sets
    train_set, val_set, test_set = split_dataset(dataset)
    
    # set the transform for validation and testing sets
    val_set.dataset.transform = val_test_transform
    test_set.dataset.transform = val_test_transform
    
    # create data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, val_loader, test_loader

# model definition
def initialize_model(num_classes=4):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# model training
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print("Start working on epoch {}".format(epoch + 1))
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        val_loss = evaluate_model(model, val_loader, criterion)  # evaluate on validation set after each epoch

# model evaluation
def evaluate_model(model, loader, criterion, mode="Validation"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if mode == "Test":
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    print(f'{mode} Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

    if mode == "Test":
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        print("Confusion Matrix:")
        print(cm)

        # Calculate precision, recall, and F1-score
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')

        # Save confusion matrix plots
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=loader.dataset.dataset.classes, yticklabels=loader.dataset.dataset.classes)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

# save the model
def save_model(model, model_path):
    # check if the model_path is a directory
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "model.pt")
    elif not model_path.endswith(".pt"):
        model_path = model_path + ".pt"
    torch.save(model.state_dict(), model_path)

def save_label_mapping(dataset, file_path):
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(file_path, 'w') as f:
        for idx, class_name in idx_to_class.items():
            f.write(f"{idx},{class_name}\n")

# main function
def main(data_dir, model_path, num_classes=4, num_epochs=10, batch_size=32, learning_rate=0.001):
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size)
    model = initialize_model(num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    print("Testing model on test dataset:")
    evaluate_model(model, test_loader, criterion, mode="Test")

    if model_path:
        save_model(model, model_path)
        # Save label mapping
        if not os.path.isdir(model_path):
            model_dir = os.path.dirname(model_path)
        save_label_mapping(train_loader.dataset.dataset, os.path.join(model_dir, "label_mapping.txt"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=False, help='Path to dataset directory')
    parser.add_argument('-m', '--model_path', type=str, required=False, help='Path to save the model')
    args = parser.parse_args()
    data_dir = args.data_dir or "./dataset"
    model_path = args.model_path
    main(data_dir, model_path)
