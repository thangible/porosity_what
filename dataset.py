import pandas as pd
from torch.utils.data import Dataset
import cv2
import os

# Define the path to the image directory and CSV file
image_dir = os.path.join('..', 'data')
csv_path = os.path.join('..', 'data', 'image_data.txt')

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_path, header=None, names=["path", "name", "magnification", "porosity"])[:8]
df["path"] = df["path"].str.replace("\\", "/", regex=False)

class PorosityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["image"], item["label"]  # Return image and label as a tuple

def process_data(df, image_dir, dimensions=(224, 224)):
    data = []
    for index, row in df.iterrows():
        print(index)
        image_path = os.path.join(image_dir, row["path"])
        try:
            # Use cv2 to read the image
            image = cv2.imread(image_path)
            # Resize the image to 224x224
            image = cv2.resize(image, dimensions)
            # Convert from BGR to RGB (as expected by most models)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convert image to float32 and normalize to range [0, 1]
            image = image.astype('float32') / 255.0
            # Rearrange the image to channel-first format (C, H, W)
            image = image.transpose(2, 0, 1)
            if row["porosity"] <= 0.05:
                label = 0
            elif row["porosity"] <= 0.10:
                label = 1
            elif row["porosity"] <= 0.20:
                label = 2
            else:
                label = 3
            data.append({"image": image, "label": label})
        except FileNotFoundError:
            print(f"File not found: {image_path}")
            continue

    dataset = PorosityDataset(data)
    return dataset

# dataset = process_data(df, image_dir)
# print(len(dataset))
# print(dataset[0])