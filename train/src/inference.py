import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from dataset_class import LicensePlateDataset
from crnn_class import CRNN, IDX2CHAR

# ==========================================
# 1. CTC Greedy Decoder
# ==========================================


def decode_predictions(predictions):
    """
    Decodes the raw output from the CRNN into a readable string.
    Removes consecutive duplicates and blank tokens.
    """
    # predictions shape: (Sequence Length, Batch Size, Num Classes)
    # Get the index of the highest probability class per time step
    _, max_indices = torch.max(predictions, 2)

    # Squeeze out the batch dimension (assuming batch_size=1)
    max_indices = max_indices.squeeze(1)

    decoded_str = []
    prev_idx = -1

    for idx in max_indices.tolist():
        # Only append if it's not a blank (0) and not a consecutive duplicate
        if idx != 0 and idx != prev_idx:
            decoded_str.append(IDX2CHAR[idx])
        prev_idx = idx

    return ''.join(decoded_str), max_indices.tolist()


# ==========================================
# 2. Inference Execution
# ==========================================


def run_inference(model_path, img_dir, num_samples=5):
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Running inference on device: {device}\n')

    # Load model
    model = CRNN().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Define visual transforms (must match training)
    transform = transforms.Compose(
        [
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Grab a few images from the directory
    img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))[:num_samples]

    if not img_paths:
        print(f'No JPG images found in {img_dir}')
        return

    print(f'{"Actual":<15} | {"Predicted":<15} | {"Raw Sequence"}')
    print('-' * 65)

    with torch.no_grad():
        for img_path in img_paths:
            # Get actual label from filename
            actual_label = os.path.basename(img_path).replace('.jpg', '')

            # Load and transform image
            image = Image.open(img_path).convert('L')
            image_tensor = (
                transform(image).unsqueeze(0).to(device)
            )  # Add batch dimension -> (1, 1, 32, 128)

            # Forward pass
            output = model(image_tensor)

            # Decode
            predicted_label, raw_indices = decode_predictions(output)

            # Create a string representation of the raw sequence for visualization
            raw_seq_str = ''.join([IDX2CHAR[idx] for idx in raw_indices])

            print(f'{actual_label:<15} | {predicted_label:<15} | {raw_seq_str[:30]}...')


if __name__ == '__main__':
    base_dir = Path(__file__).parent.resolve()

    # Update these paths if your checkpoint or data folder is named differently
    MODEL_CHECKPOINT = base_dir / 'crnn_plate_epoch_50.pth'
    DATA_DIR = base_dir / '..' / 'data' / 'val'

    if not MODEL_CHECKPOINT.exists():
        print(f'Checkpoint not found at: {MODEL_CHECKPOINT}')
    else:
        run_inference(MODEL_CHECKPOINT, DATA_DIR, num_samples=10)
