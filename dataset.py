import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision.transforms import Compose, ToTensor, Normalize, Resize



# Custom Dataset class
class PorosityDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.image_dir, row["path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            raise

        if self.transform:
            image = self.transform(image)

        porosity = row["porosity"]
        if porosity <= 0.05:
            label = 0
        elif porosity <= 0.10:
            label = 1
        elif porosity <= 0.20:
            label = 2
        else:
            label = 3

        return image, label

