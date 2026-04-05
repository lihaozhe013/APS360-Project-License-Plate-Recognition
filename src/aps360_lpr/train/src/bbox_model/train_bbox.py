import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from aps360_lpr.train.src.bbox_model.bbox_cnn import BoundingBoxCNN
from pathlib import Path
from tqdm import tqdm

class LicensePlateDataset(Dataset):
    def __init__(self, data_dir, image_size=(224, 224), transform=None):
        self.data_dir = Path(data_dir)
        self.file_list = list(self.data_dir.glob('embedded_*.jpg'))
        self.image_size = image_size
        self.transform = transform
        
        # Format: embedded_x1_y1_x2_y2_x3_y3_x4_y4_idx.jpg
        # Example: embedded_2830.3_2652.4_3241.0_2599.3_2810.7_2879.4_3214.4_2816.0_0.jpg
        self.samples = []
        for file_path in self.file_list:
            filename = file_path.name
            parts = filename.split('_')
            try:
                # Extract coordinates from filenames
                coords = [float(p) for p in parts[1:9]]
                self.samples.append((file_path, coords))
            except Exception as e:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, coords = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w, _ = image.shape
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Normalize coordinates to [0, 1] range based on original image size
        norm_coords = []
        for i in range(0, 8, 2):
            norm_coords.append(coords[i] / w)      # x
            norm_coords.append(coords[i+1] / h)    # y
            
        norm_coords = torch.tensor(norm_coords, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            
        return image, norm_coords

def train_model(data_dir, epochs=10, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LicensePlateDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        print("No valid images found in directory.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = BoundingBoxCNN(pretrained=True).to(device)
    criterion = nn.L1Loss() # L1Loss (Mean Absolute Error) is better for pixel coordinates vs MSE
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training for {epochs} epochs on {len(dataset)} samples...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1} Loss (L1): {epoch_loss:.6f}")
        
    # Save the trained model
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/bbox_model.pth")
    print("Model saved to weights/bbox_model.pth")

if __name__ == "__main__":
    # Adjust this path based on where your script is run from
    DATA_DIR = "../../../../../dataset/background_embedder_out_"
    train_model(DATA_DIR, epochs=10, batch_size=16)
