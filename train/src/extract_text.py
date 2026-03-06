import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ==========================================
# 1. Dataset & Collate Function
# ==========================================

# Standard license plate characters (36 total).
# Index 0 is reserved for the CTC "blank" token.
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank token


class LicensePlateDataset(Dataset):
    def __init__(self, img_dir, img_width=128, img_height=32):
        self.img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))

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
        image = Image.open(img_path).convert("L")

        # Extract label from filename (e.g., "ABC1234.jpg" -> "ABC1234")
        label_str = os.path.basename(img_path).replace(".jpg", "")

        # Apply visual transforms
        image = self.transform(image)

        # Convert string label to list of integers
        # Filter out characters not in CHARS, like the hyphen in "ABC-123"
        label = [CHAR2IDX[c] for c in label_str if c in CHAR2IDX]

        return image, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """
    Custom collate function for CTC loss.
    CTC needs targets as a 1D tensor of concatenated labels and a tensor of target lengths.
    """
    images, labels = zip(*batch)
    images = torch.stack(images, 0)

    # Calculate lengths of each sequence in the batch
    target_lengths = torch.tensor([len(t) for t in labels], dtype=torch.long)

    # Concatenate all labels into a 1D tensor
    targets = torch.cat(labels)

    return images, targets, target_lengths


# ==========================================
# 2. CRNN Model Architecture
# ==========================================


class CRNN(nn.Module):
    def __init__(self, img_channel=1, num_classes=NUM_CLASSES, hidden_size=256):
        super(CRNN, self).__init__()

        # CNN Backbone
        # Input shape: (Batch, 1, 32, 128)
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (B, 64, 16, 64)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (B, 128, 8, 32)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # Pool only height, keep width (sequence length)
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Output: (B, 256, 4, 32)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Output: (B, 512, 2, 32)
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # Output: (B, 512, 1, 31)
        )

        # RNN Backbone
        self.rnn = nn.GRU(
            512, hidden_size, num_layers=2, bidirectional=True, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 1. Feature extraction
        conv_out = self.cnn(x)

        # 2. Map to sequence:
        # conv_out shape is (Batch, Channels, Height, Width) -> (B, 512, 1, 31)
        # Squeeze the height dimension (which is 1)
        conv_out = conv_out.squeeze(2)  # (B, 512, 31)

        # Transpose to (Batch, SequenceLength, Channels) for RNN
        conv_out = conv_out.permute(0, 2, 1)  # (B, 31, 512)

        # 3. Sequence modeling
        rnn_out, _ = self.rnn(conv_out)  # (B, 31, 512)

        # 4. Classification
        output = self.fc(rnn_out)  # (B, 31, NUM_CLASSES)

        # PyTorch CTC Loss expects input of shape (SequenceLength, Batch, NumClasses)
        output = output.permute(1, 0, 2)

        # Apply log softmax as required by CTCLoss
        output = F.log_softmax(output, dim=2)
        return output


# ==========================================
# 3. Training Loop
# ==========================================


def train():
    base_dir = Path(__file__).parent.resolve()

    # --- Configurations ---
    img_dir = base_dir / ".." / "data"
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Training on device: {device}")

    # --- Data Loading ---
    dataset = LicensePlateDataset(img_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # --- Initialization ---
    model = CRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # zero_infinity=True prevents loss from spiking to inf if target is longer than model prediction
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # --- Training ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, targets, target_lengths) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            # log_probs shape: (Sequence_Length, Batch_Size, Num_Classes)
            log_probs = model(images)

            # Input lengths (Sequence Length of the predictions)
            # The CNN architecture outputs a sequence length of 31 for W=128
            input_lengths = torch.full(
                size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long
            )

            # Calculate CTC Loss
            # Workaround: CTCLoss is not implemented for MPS, run on CPU if needed
            if device.type == "mps":
                loss = criterion(
                    log_probs.cpu(),
                    targets.cpu(),
                    input_lengths.cpu(),
                    target_lengths.cpu(),
                )
            else:
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # Backward pass & optimization
            loss.backward()

            # Gradient clipping to prevent exploding gradients in RNNs
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"crnn_plate_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train()
