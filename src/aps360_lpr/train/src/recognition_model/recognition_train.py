import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from aps360_lpr.train.src.recognition_model.recognition_dataset_class import LicensePlateDataset
from aps360_lpr.train.src.recognition_model.crnn_class import CRNN

# ==========================================
# 1. Dataset & Collate Function
# ==========================================


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


def train():
    base_dir = Path(__file__).parent.resolve() / '..'

    # --- Configurations ---
    train_dir = base_dir / 'data' / 'train'
    val_dir = base_dir / 'data' / 'val'
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Training on device: {device}')

    # --- Data Loading ---
    train_dataset = LicensePlateDataset(train_dir)
    val_dataset = LicensePlateDataset(val_dir)

    print(f'Training Limit: {len(train_dataset)}')
    print(f'Validation Limit: {len(val_dataset)}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # --- Initialization ---
    model = CRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # zero_infinity=True prevents loss from spiking to inf if target is longer than model prediction
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Lists to store losses
    train_losses = []
    val_losses = []

    # --- Training ---
    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0

        for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
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
            if device.type == 'mps':
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

            train_loss_epoch += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}'
                )

        avg_train_loss = train_loss_epoch / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'--- Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f} ---')

        # --- Validation ---
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                log_probs = model(images)

                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=log_probs.size(0),
                    dtype=torch.long,
                )

                if device.type == 'mps':
                    loss = criterion(
                        log_probs.cpu(),
                        targets.cpu(),
                        input_lengths.cpu(),
                        target_lengths.cpu(),
                    )
                else:
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                val_loss_epoch += loss.item()

        avg_val_loss = val_loss_epoch / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'--- Epoch {epoch + 1} Val Loss: {avg_val_loss:.4f} ---')

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'crnn_plate_epoch_{epoch + 1}.pth')

    # --- Summary & Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('CTC Loss')
    plt.legend()
    plt.grid(True)
    plot_path = base_dir / 'loss_plot.png'
    plt.savefig(plot_path)
    print(f'Loss plot saved to {plot_path}')

    summary_path = base_dir / 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('Training Summary\n')
        f.write('================\n')
        f.write(f'Total Epochs: {epochs}\n')
        f.write(f'Training Set Size: {len(train_dataset)}\n')
        f.write(f'Validation Set Size: {len(val_dataset)}\n')
        f.write(f'Final Train Loss: {train_losses[-1]:.4f}\n')
        f.write(f'Final Val Loss: {val_losses[-1]:.4f}\n')
        f.write(f'Best Val Loss: {min(val_losses):.4f}\n')
    print(f'Training summary saved to {summary_path}')


if __name__ == '__main__':
    train()
