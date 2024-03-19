import torch
import torch.nn as nn
import snntorch as snn

class SegformerClassification(nn.Module):
    def __init__(self, num_classes, input_dim):
        super().__init__()
        # Classification head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x=torch.cat(x, dim=1)

        # Global average pooling
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        # Classification
        x = self.fc(x)

        return x