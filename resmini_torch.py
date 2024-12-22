# import
import dicom2nifti
import nibabel as nib
import nilearn as nil
from nilearn import plotting
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.transforms import functional as tf
import random
from torchvision import transforms

# path defined
images_path = '/home/sohinim/Brain_Age_Estimation/ResNet-10_Reimplementation/data_gz'
target_path = '/home/sohinim/Brain_Age_Estimation/ResNet-10_Reimplementation/participants.tsv'

# preprocess labels
labels_all = pd.read_csv(target_path, sep = '\t')
labels = (labels_all)['AgeGroup']

# string labels
string_labels = labels 
label_to_index = {'4yo': 0, '3yo': 0, '5yo': 0, '7yo': 1, '8-12yo': 1, 'Adult': 2}
index_to_label = {0: 'Adults', 1: 'Ages 3-5', 2: 'Ages 7-12'}

def resize_images(single_image_data, target_shape=(95, 79)):
    zoom_factors = (target_shape[0] / single_image_data.shape[0], target_shape[1] / single_image_data.shape[1])
    resized_image = ndi.zoom(single_image_data, zoom_factors, order=1)  # Bilinear interpolation
    return resized_image

# preprocessing
class Augmentations:
    def __init__(self, max_shift=10):
        self.max_shift = max_shift 
    def __call__(self, image):
        x_shift = random.randint(-self.max_shift, self.max_shift)
        y_shift = random.randint(-self.max_shift, self.max_shift)
        shifted_image = tf.affine(image, angle=0, translate=(x_shift, y_shift), scale=1.0, shear=0) # shift
        flipped_image = tf.hflip(shifted_image) # Random horizontal flip
        return shifted_image, flipped_image

transform = transforms.Compose([
    transforms.ToTensor(),          
    Augmentations(max_shift=10)     
])


class BrainAgeDataset(Dataset):
    def __init__(self, image_dir, labels, label_mapping, transform=None):
        self.file_paths = sorted(glob.glob(os.path.join(image_dir, 'sub-pixar*_T1w.nii.gz')))
        self.labels = labels
        self.label_mapping = label_mapping
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((176, 176)),  # Set to the desired fixed size
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_data = nib.load(file_path).get_fdata()
        slice_idx = image_data.shape[2] // 2
        image = image_data[:, :, slice_idx]
        image = np.array(image, dtype=np.float32)
        image = self.transform(image)
        label_str = self.labels[idx]
        label = self.label_mapping[label_str]
        return image, label


# Define image directory and labels
image_dir = '/home/sohinim/Brain_Age_Estimation/ResNet-10_Reimplementation/data_gz'  # Path to the folder with .nii.gz files
labels = labels.to_list() #labels as list


# Add function to plot and save training/validation accuracy and loss
def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("tv_loss.png")
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Training and Validation Accuracy")
    plt.savefig("tv_accuracy.png")
    plt.show()

# Update batch size and epochs
batch_size = 24
num_epochs = 200

dataset = BrainAgeDataset(image_dir=image_dir, labels=labels, label_mapping=label_to_index, transform=None)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# convolutional block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle changes in dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # First convolution + ReLU
        out = self.bn2(self.conv2(out))  # Second convolution
        out += self.shortcut(x)  # Adding the shortcut
        out = F.relu(out)  # ReLU after addition
        return out

# ResMini Model with stacked residual blocks
class ResMini(nn.Module):
    def __init__(self, num_classes=3):
        super(ResMini, self).__init__()
        self.in_channels = 8

        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(8)

        # Residual Blocks (stacked in layers)
        self.layer1 = self._make_layer(8, stride=1, num_blocks=2)  # Two blocks in layer 1
        self.layer2 = self._make_layer(16, stride=2, num_blocks=2)  # Two blocks in layer 1
        self.layer3 = self._make_layer(32, stride=2, num_blocks=2)  # Two blocks in layer 2
        self.layer4 = self._make_layer(64, stride=2, num_blocks=2)  # Two blocks in layer 3

        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, stride, num_blocks):
        # Create a sequence of residual blocks (num_blocks per layer)
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            stride = 1  # After the first block in the layer, the stride stays 1 for subsequent blocks
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # First convolution + ReLU
        out = self.layer1(out)
        out = self.layer2(out)  
        out = self.layer3(out) 
        out = self.layer4(out)  
        out = F.adaptive_avg_pool2d(out, (1, 1))  # Global average pooling
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)  # Fully connected layer
        return out
    
    
# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Further split train into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders with batch size 24
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

# Track metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

model = ResMini()  
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.00002)

def train(model, train_loader, criterion, optimizer, epoch):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Ensure the model is on the correct device
    
    train_loss, correct, total = 0.0, 0, 0
    correct = 0
    total = 0
    model.train()
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy)
    
    accuracy = 100 * correct / total
    print(f'Epoch [{epoch}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_accuracy = 100 * correct / total
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.backends.cudnn.enabled = False

# Training and Validation
num_epochs = 200
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    validate(model, val_loader, criterion)
    
plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies)


def test(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for faster evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the same device as model
            outputs = model(images)
            loss = criterion(outputs, labels)  # Calculate loss
            test_loss += loss.item() * images.size(0)  # Accumulate the loss

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            correct += (predicted == labels).sum().item()  # Count correct predictions
            total += labels.size(0)  # Count total samples

    # Compute average loss and accuracy
    avg_loss = test_loss / total
    accuracy = correct / total * 100
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")


# Set up the test data loader
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)  # Adjust batch size if necessary

# Run the testing loop
criterion = nn.CrossEntropyLoss()  # Same criterion as used during training
test(model, test_loader, criterion)
