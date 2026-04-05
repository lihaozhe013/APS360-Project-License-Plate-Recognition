import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from aps360_lpr.train.src.recognition_model.crnn_class import CHAR2IDX


class LicensePlateDataset(Dataset):
    def __init__(self, img_dir, img_width=128, img_height=32):
        self.img_paths = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))

        # Resize to fixed height of 32 and width of 128, convert to tensor, normalize
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # Load as grayscale
        image = Image.open(img_path).convert('L')

        # Extract label from filename (e.g., "ABC1234.jpg" -> "ABC1234")
        label_str = os.path.basename(img_path).replace('.jpg', '')

        # Apply visual transforms
        image = self.transform(image)

        # Convert string label to list of integers
        # Filter out characters not in CHARS, like the hyphen in "ABC-123"
        label = [CHAR2IDX[c] for c in label_str if c in CHAR2IDX]

        return image, torch.tensor(label, dtype=torch.long)
