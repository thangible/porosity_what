import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from dataset import PorosityDataset
from torchvision.transforms import ToPILImage
import pandas as pd
import wandb
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score

# Training and validation loop
num_epochs = 100
dimensions = (1024,1024)
batch_size = 8



# Define the path to the image directory and CSV file
image_dir = os.path.join('..', 'data')
csv_path = os.path.join('..', 'data', 'image_data.txt')

# Define transformations for the images
transform = Compose([
    Resize(dimensions),
    ToTensor(),                 
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_path, header=None, names=["path", "name", "magnification", "porosity"])
df["path"] = df["path"].str.replace("\\", "/", regex=False)



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
# model = resnet50(weights=ResNet50_Weights.DEFAULT)
# model = resnet152(weights=ResNet152_Weights.DEFAULT)
model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)

# Modify the final fully connected layer to match the number of classes (4 in this case)
num_classes = 1
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use MSE loss for regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

import time  # để đo thời gian mỗi epoch

# Initialize W&B
wandb.init(
    project="resnet50-porosity",
    config={
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "model": "ResNet50",
        "image_size": dimensions,
    }
)
for epoch in range(num_epochs):
    start_time = time.time()

    # -------- TRAINING --------
    model.train()
    train_loss, train_mae = 0.0, 0.0
    train_preds, train_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.extend(outputs.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    # print(train_preds, train_labels )
    train_loss_avg = train_loss / len(train_loader)
    train_mae_avg = mean_absolute_error(train_labels, train_preds)
    train_r2 = r2_score(train_labels, train_preds)

    # -------- VALIDATION --------
    model.eval()
    val_loss, val_mae = 0.0, 0.0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    # print("Sample val_labels:", val_labels[:5])
    # print("Sample val_preds:", val_preds[:5])
    # print("Any NaNs in val_labels?", np.isnan(val_labels).any())
    # print("Any NaNs in val_preds?", np.isnan(val_preds).any())
    val_loss_avg = val_loss / len(val_loader)
    val_mae_avg = mean_absolute_error(val_labels, val_preds)
    val_r2 = r2_score(val_labels, val_preds)
    
    if epoch % 10 == 0:  # or just once at the end
        # Log up to 8 sample images
        num_samples = min(8, len(val_dataset))
        sample_imgs, sample_labels = zip(*[val_dataset[i] for i in range(num_samples)])
        sample_imgs_tensor = torch.stack(sample_imgs).to(device)
        model.eval()
        with torch.no_grad():
            preds = model(sample_imgs_tensor).squeeze().cpu().numpy()

        # Prepare image logs
        table = wandb.Table(columns=["Image", "True Porosity", "Predicted Porosity"])
        to_pil = ToPILImage()

        for i in range(num_samples):
            img = to_pil(sample_imgs[i].cpu())
            true_val = round(float(sample_labels[i].item()), 4)
            pred_val = round(float(preds[i]), 4)

            table.add_data(wandb.Image(img), true_val, pred_val)

        wandb.log({"Sample Predictions": table})


    # -------- LOGGING --------
    epoch_duration = time.time() - start_time
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss_avg:.4f}, MAE: {train_mae_avg:.4f}, R²: {train_r2:.4f} | "
          f"Val Loss: {val_loss_avg:.4f}, MAE: {val_mae_avg:.4f}, R²: {val_r2:.4f} | "
          f"Time: {epoch_duration:.1f}s")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss_avg,
        "train_mae": train_mae_avg,
        "train_r2": train_r2,
        "val_loss": val_loss_avg,
        "val_mae": val_mae_avg,
        "val_r2": val_r2,
        "epoch_time_sec": epoch_duration,
        "learning_rate": optimizer.param_groups[0]['lr']
    })
    
        # Create a wandb Table with your data
    table = wandb.Table(data=[[x, y] for x, y in zip(val_labels, val_preds)],
                        columns=["True", "Predicted"])

    # Log the scatter plot
    wandb.log({
        "Validation: True vs Predicted": wandb.plot.scatter(
            table,
            "True",
            "Predicted",
            title="Validation: True vs Predicted Porosity"
        )
    })

wandb.run.define_metric("epoch")
wandb.run.define_metric("train_*", step_metric="epoch")
wandb.run.define_metric("val_*", step_metric="epoch")

# Group plots logically in dashboard
wandb.define_metric("train_loss", summary="min")
wandb.define_metric("val_loss", summary="min")
wandb.define_metric("train_r2", summary="max")
wandb.define_metric("val_r2", summary="max")
