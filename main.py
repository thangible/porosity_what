import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from dataset import PorosityDataset
from torchvision.transforms import ToPILImage
import pandas as pd
# import wandb
import os

# Define the path to the image directory and CSV file
image_dir = os.path.join('..', 'data')
csv_path = os.path.join('..', 'data', 'image_data.txt')

# Define transformations for the images
transform = Compose([
    Resize((224, 224)),
    ToTensor(),                 
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_path, header=None, names=["path", "name", "magnification", "porosity"])[:4]
df["path"] = df["path"].str.replace("\\", "/", regex=False)

# Training and validation loop
num_epochs = 100
dimensions = (224, 224)
batch_size = 8


# Set seed for reproducibility
generator = torch.Generator().manual_seed(42)

# Process the dataset
dataset = PorosityDataset(df, image_dir, transform=transform)

# Split the dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Load the pretrained ResNet50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the final fully connected layer to match the number of classes (4 in this case)
num_classes = 1
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use MSE loss for regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import time  # để đo thời gian mỗi epoch

# Initialize W&B
# wandb.init(
#     project="resnet50-porosity",
#     config={
#         "epochs": num_epochs,
#         "batch_size": batch_size,
#         "learning_rate": 0.001,
#         "model": "ResNet50",
#         "image_size": dimensions,
#     }
# )

for epoch in range(num_epochs):
    start_time = time.time()

    # ---------------- TRAINING ----------------
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
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

    train_loss_avg = train_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    # ---------------- VALIDATION ----------------
    model.eval()
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

    val_loss_avg = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    # ---------------- LOGGING ----------------
    epoch_duration = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss_avg:.4f}, Acc: {train_accuracy:.2f}% | "
          f"Val Loss: {val_loss_avg:.4f}, Acc: {val_accuracy:.2f}% | Time: {epoch_duration:.1f}s")

    # wandb.log({
    #     "epoch": epoch + 1,
    #     "train_loss": train_loss_avg,
    #     "train_accuracy": train_accuracy,
    #     "val_loss": val_loss_avg,
    #     "val_accuracy": val_accuracy,
    #     "epoch_time_sec": epoch_duration,
    #     "learning_rate": optimizer.param_groups[0]['lr']
    # })
