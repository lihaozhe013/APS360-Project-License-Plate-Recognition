import torch
import torch.nn as nn
import torchvision.models as models

class BoundingBoxCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(BoundingBoxCNN, self).__init__()
        # Load a pretrained ResNet18 model
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # We need to output 8 coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
        # Modify the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 8)

    def forward(self, x):
        # We can add a Sigmoid activation if we normalize our targets to [0, 1]
        # Or leave it linear if we predict absolute pixel coordinates (less common but possible)
        # Better practice: Normalize target coordinates to [0, 1] and use Sigmoid
        x = self.resnet(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    # Test the model with dummy data
    model = BoundingBoxCNN()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: 1, 8)")
    print(output)
