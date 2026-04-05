import torch.nn as nn
import torch.nn.functional as F

CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
IDX2CHAR[0] = '-'
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank token


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
        b, c, h, w = conv_out.size()
        assert h == 1, "the height of conv must be 1"
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
